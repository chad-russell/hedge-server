extern crate bincode;
extern crate chrono;
extern crate rand;
extern crate serde_derive;
extern crate serde_json;
extern crate statrs;

use std::collections::HashMap;
use std::io::Read;
use std::str;

use actix_web::{get, post, web, App, HttpRequest, HttpResponse, HttpServer, Responder};

use chrono::{Datelike, NaiveDate};

use serde::{Deserialize, Serialize};

use statrs::distribution::{Continuous, Normal, Univariate};

#[actix_rt::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .service(option_chain_for_symbol_and_date)
            .service(find_delta)
            .service(backtest)
    })
    .bind("127.0.0.1:8000")?
    .run()
    .await
}

#[derive(Debug, Serialize, Deserialize)]
struct BacktestRequest {
    underlying: String,
    legs: Vec<BacktestRequestLeg>,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
struct BacktestRequestLeg {
    is_call: bool,
    is_buy: bool,
    delta: f32,
    dte: i32,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct BacktestResponse {
    trials: Vec<BacktestTrialResponse>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BacktestTrialResponse {
    leg: BacktestRequestLeg,
    start_date: String,
    expiration_date: String,
    pl: Vec<f32>,
}

#[post("/backtest")]
async fn backtest(req: web::Json<BacktestRequest>) -> impl Responder {
    let mut response = BacktestResponse::default();

    let mut expiration_dates: Vec<String> = Vec::new();
    let mut start_dates: Vec<Vec<String>> = Vec::new();

    for year in 2005..=2020 {
        let decoded = read_db_file(&year.to_string(), &req.underlying);
        if decoded.is_none() {
            return HttpResponse::Ok().json(format!(
                "Failed to read db file for {} {}",
                year, req.underlying
            ));
        }
        let decoded = decoded.unwrap();
        let mut day_to_check = NaiveDate::from_ymd(year, 1, 1).num_days_from_ce();
        while !decoded.contains_key(&day_to_check) {
            day_to_check += 1;
        }

        loop {
            if !decoded.contains_key(&day_to_check) {
                break;
            }

            let mut ed = Vec::new();
            for c in decoded[&day_to_check].contracts.iter() {
                let exp_date = NaiveDate::from_num_days_from_ce(c.expiration_date).to_string();
                if !ed.contains(&exp_date) {
                    ed.push(exp_date);
                }
            }

            ed.sort();

            let mut next_day_to_check = NaiveDate::parse_from_str(&ed[0], "%Y-%m-%d")
                .unwrap()
                .num_days_from_ce();
            if next_day_to_check == day_to_check {
                next_day_to_check = NaiveDate::parse_from_str(&ed[1], "%Y-%m-%d")
                    .unwrap()
                    .num_days_from_ce();
            }
            day_to_check = next_day_to_check;

            let mut limit = 0;
            while !decoded.contains_key(&day_to_check) && limit < 365 {
                day_to_check += 1;
                limit += 1;
            }

            expiration_dates.extend(ed);
        }
    }

    expiration_dates.sort();
    expiration_dates.dedup();

    start_dates = req
        .legs
        .iter()
        .map(|leg| {
            expiration_dates
                .iter()
                .map(|d| {
                    let date = NaiveDate::parse_from_str(d, "%Y-%m-%d")
                        .unwrap()
                        .num_days_from_ce();
                    let date = date - leg.dte; // TODO(chad): configurable number of days until expiration
                    NaiveDate::from_num_days_from_ce(date).to_string()
                })
                .collect()
        })
        .collect();

    let mut db_cache = DbCache::default();

    let mut aggregated_legs = Vec::new();

    for (leg, dates_list) in req.legs.iter().zip(start_dates.iter()) {
        for (date_index, (start_date, expiration_date)) in
            dates_list.iter().zip(expiration_dates.iter()).enumerate()
        {
            let start_date = NaiveDate::parse_from_str(start_date, "%Y-%m-%d").unwrap();
            let start_date_days = start_date.num_days_from_ce();

            let expiration_date = NaiveDate::parse_from_str(expiration_date, "%Y-%m-%d").unwrap();
            let expiration_date_days = expiration_date.num_days_from_ce();

            println!(
                "Running simulation for leg: {:?}, start_date: {}, expiration_date: {}",
                leg,
                start_date,
                NaiveDate::from_num_days_from_ce(expiration_date_days)
            );

            let mut ag_leg_value = Vec::new();

            let strike = db_cache
                .entry_closest_or_before(start_date.num_days_from_ce(), &req.underlying)
                .map(|db_entry| {
                    if leg.is_call {
                        find_call_strike_by_delta(&db_entry, leg.delta, start_date, expiration_date)
                    } else {
                        find_put_strike_by_delta(&db_entry, leg.delta, start_date, expiration_date)
                    }
                });

            match strike {
                Some(strike) => {
                    let mut trial = BacktestTrialResponse {
                        leg: *leg,
                        start_date: start_date.to_string(),
                        expiration_date: expiration_date.to_string(),
                        pl: Vec::new(),
                    };

                    for date_days in start_date_days..expiration_date_days {
                        // Look up the value of the leg on the day
                        let date = NaiveDate::from_num_days_from_ce(date_days);

                        let contract = db_cache
                            .entry(date.num_days_from_ce(), &req.underlying)
                            .map(|db_entry| {
                                if leg.is_call {
                                    find_call_contract_by_strike(
                                        &db_entry,
                                        strike,
                                        date,
                                        expiration_date,
                                    )
                                } else {
                                    find_put_contract_by_strike(
                                        &db_entry,
                                        strike,
                                        date,
                                        expiration_date,
                                    )
                                }
                            });

                        if let Some(Some(contract)) = contract {
                            trial.pl.push(contract.last);
                        }
                    }

                    response.trials.push(trial);
                }
                None => (),
            }

            aggregated_legs.push(ag_leg_value);
        }
    }

    HttpResponse::Ok().json(response)
}

#[derive(Default)]
struct DbCache {
    // year -> { date, DbEntry }
    values: HashMap<i32, HashMap<i32, DbEntry>>,
}

impl DbCache {
    fn entry(&mut self, date_days: i32, symbol: &str) -> Option<&DbEntry> {
        let date = NaiveDate::from_num_days_from_ce(date_days);

        let found_db_file = self.values.get(&date.year()).is_some();
        if !found_db_file {
            match read_db_file(&date.year().to_string(), symbol) {
                Some(val) => {
                    self.values.insert(date.year(), val);
                }
                None => {
                    return None;
                }
            }
        }

        self.values.get(&date.year())?.get(&date_days)
    }

    fn entry_closest_or_before(&mut self, date_days: i32, symbol: &str) -> Option<&DbEntry> {
        let date = NaiveDate::from_num_days_from_ce(date_days);
        let found_db_file = self.values.get(&date.year()).is_some();
        if !found_db_file {
            match read_db_file(&date.year().to_string(), symbol) {
                Some(val) => {
                    self.values.insert(date.year(), val);
                }
                None => {
                    return None;
                }
            }
        }
        let db_file = self.values.get(&date.year())?;

        let mut limit = 0;
        while limit <= 10 {
            match db_file.get(&(date_days - limit)) {
                Some(result) => {
                    return Some(result);
                }
                None => {
                    limit += 1;
                }
            }
        }

        None
    }
}

#[get("/{symbol}/delta/{delta}")]
async fn find_delta(req: HttpRequest) -> impl Responder {
    let symbol = req.match_info().get("symbol").unwrap();
    let delta: f32 = req.match_info().get("delta").unwrap().parse().unwrap();

    let sim_date = NaiveDate::from_ymd(2018, 7, 9);

    // read all symbols for this year
    let decoded = read_db_file(&sim_date.year().to_string(), symbol);
    if decoded.is_none() {
        return format!("Failed to read db file");
    }
    let decoded = decoded.unwrap();
    let decoded = &decoded[&sim_date.num_days_from_ce()];

    let exp_date = NaiveDate::from_num_days_from_ce(decoded.contracts[0].expiration_date);

    let call_strike = find_call_strike_by_delta(&decoded, delta, sim_date, exp_date);
    let put_strike = find_put_strike_by_delta(&decoded, delta, sim_date, exp_date);

    format!(
        "The price of {} on {} for the {} expiration cycle was: {}. The {} delta call option's strike price is: {}, and the {} delta put option's strike price is: {}\n",
        symbol, sim_date, exp_date, decoded.underlying_price, delta, call_strike, delta, put_strike,
    )
}

fn find_call_strike_by_delta(
    entry: &DbEntry,
    delta: f32,
    sim_date: NaiveDate,
    exp_date: NaiveDate,
) -> f32 {
    let mut strike = std::f32::INFINITY;

    for contract in entry.contracts.iter() {
        let cr = ContractResponse::from(contract, entry.underlying_price, sim_date);

        if cr.is_call
            && cr.delta <= delta / 100.0
            && cr.expiration_date == exp_date
            && cr.strike < strike
        {
            strike = cr.strike;
        }
    }

    strike
}

fn find_put_strike_by_delta(
    entry: &DbEntry,
    delta: f32,
    sim_date: NaiveDate,
    exp_date: NaiveDate,
) -> f32 {
    let mut strike = std::f32::NEG_INFINITY;
    for contract in entry.contracts.iter() {
        let cr = ContractResponse::from(contract, entry.underlying_price, sim_date);

        if !cr.is_call
            && cr.delta <= delta / 100.0
            && cr.expiration_date == exp_date
            && cr.strike > strike
        {
            strike = cr.strike;
        }
    }

    strike
}

fn find_call_contract_by_strike(
    entry: &DbEntry,
    strike: f32,
    sim_date: NaiveDate,
    exp_date: NaiveDate,
) -> Option<ContractResponse> {
    entry
        .contracts
        .iter()
        .filter(|c| {
            c.is_call && c.expiration_date == exp_date.num_days_from_ce() && c.strike == strike
        })
        .map(|c| ContractResponse::from(c, entry.underlying_price, sim_date))
        .next()
}

fn find_put_contract_by_strike(
    entry: &DbEntry,
    strike: f32,
    sim_date: NaiveDate,
    exp_date: NaiveDate,
) -> Option<ContractResponse> {
    entry
        .contracts
        .iter()
        .filter(|c| {
            !c.is_call && c.expiration_date == exp_date.num_days_from_ce() && c.strike == strike
        })
        .map(|c| ContractResponse::from(c, entry.underlying_price, sim_date))
        .next()
}

#[get("/{symbol}/exp/{exp_date}/sim/{sim_date}/{strike}")]
async fn option_chain_for_symbol_and_date(req: HttpRequest) -> impl Responder {
    let symbol = req.match_info().get("symbol").unwrap();

    // let exp_date = req.match_info().get("exp_date").unwrap();
    // let exp_date = NaiveDate::parse_from_str(exp_date, "%Y-%m-%d").unwrap();

    let sim_date = req.match_info().get("sim_date").unwrap();
    let sim_date = NaiveDate::parse_from_str(sim_date, "%Y-%m-%d").unwrap();

    // let strike = req.match_info().get("strike").unwrap();

    let decoded = read_db_file(&sim_date.year().to_string(), symbol);
    if decoded.is_none() {
        return format!("Failed to read db file");
    }
    let decoded = decoded.unwrap();
    let decoded = &decoded[&sim_date.num_days_from_ce()];

    let contracts = decoded
        .contracts
        .iter()
        //.filter(|c| {
        ////c.strike == strike.parse::<f32>().unwrap()
        ////&& c.expiration_date == exp_date.num_days_from_ce()
        //})
        .map(|c| ContractResponse::from(c, decoded.underlying_price, sim_date))
        .collect::<Vec<_>>();

    format!("{}\n{:#?}\n", decoded.underlying_price, contracts)
}

fn read_db_file(year: &str, symbol: &str) -> Option<HashMap<i32, DbEntry>> {
    std::fs::File::open(&format!(
        "/Users/chadrussell/Projects/options_history/bin/{0}/{0}-{1}.bin",
        year, symbol
    ))
    .map(|mut file| {
        let mut buffer = Vec::<u8>::new();
        file.read_to_end(&mut buffer).unwrap();
        bincode::deserialize(&buffer[..]).expect("Failed to deserialize")
    })
    .ok()
}

// todo(chad): put this in another file
#[derive(Debug, Serialize, Deserialize)]
struct DbEntry {
    underlying_price: f32,
    contracts: Vec<Contract>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Contract {
    is_call: bool,
    expiration_date: i32,
    strike: f32,
    last: f32,
    bid: f32,
    ask: f32,
    open_interest: u32,
}

impl Contract {
    /// Use the Newton-Rhapson method to calculate implied volatility, using vega as the derivative
    // todo(chad): don't use dte here, accept a simulation date instead
    // fn implied_volatility(&self, spot: f32, t: f32) -> f32 {
    //     let mut vol = 0.5;

    //     let mut eps = 1.0;
    //     let tol = 0.001;

    //     let max_iter = 1000;
    //     let mut iter = 0;

    //     // if self.is_call && self.strike == 175.0 && self.ask == 207.75 {
    //     //     let stopme = 3;
    //     // }

    //     while eps > tol {
    //         iter += 1;
    //         if iter > max_iter {
    //             break;
    //         }

    //         let orig_vol = vol;

    //         let (d1, _) = d(vol, spot, self.strike, t);
    //         let function_value = call_price(vol, spot, self.strike, t) - self.last;

    //         let vega = spot * npdf(d1) * t.sqrt();
    //         vol -= function_value / vega;

    //         eps = ((vol - orig_vol) / orig_vol).abs();
    //     }

    //     if self.strike == 360.0 && self.is_call && self.expiration_date == NaiveDate::from_ymd(2020, 8, 21).num_days_from_ce() {
    //         println!("implied volatility for the one you care about: {}", vol);
    //     }

    //     // assert!(!vol.is_nan());
    //     // assert!(vol != std::f32::INFINITY);
    //     // assert!(vol != std::f32::NEG_INFINITY);

    //     vol
    // }

    fn implied_volatility(&self, spot: f32, t: f32) -> f32 {
        let mut closest_vol = 0.0;

        let mut min_eps = std::f32::INFINITY;

        let mut vol = 0.0;
        while vol < 1.0 {
            vol += 0.01;

            let theo_price = if self.is_call {
                call_price(vol, spot, self.strike, t)
            } else {
                put_price(vol, spot, self.strike, t)
            };

            let eps = (theo_price - self.last).abs();
            if eps < min_eps {
                min_eps = eps;
                closest_vol = vol;
            }
        }

        closest_vol
    }
}

fn d(sigma: f32, spot: f32, strike: f32, t: f32) -> (f32, f32) {
    // TODO(chad): incorporate risk-free rate and dividend yield
    let r = 0.0;
    let q = 0.0;

    let d1 = ((spot / strike).ln() + t * (r - q + (sigma * sigma / 2.0))) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();

    (d1, d2)
}

fn call_price(sigma: f32, spot: f32, strike: f32, t: f32) -> f32 {
    // TODO(chad): incorporate risk-free rate and dividend yield
    let r = 0.0;
    let q = 0.0;

    let (d1, d2) = d(sigma, spot, strike, t);

    spot * (-q * t).exp() * ncdf(d1) - strike * (-r * t).exp() * ncdf(d2)
}

fn put_price(sigma: f32, spot: f32, strike: f32, t: f32) -> f32 {
    // TODO(chad): incorporate risk-free rate and dividend yield
    let r = 0.0;
    let q = 0.0;

    let (d1, d2) = d(sigma, spot, strike, t);

    strike * (-r * t).exp() * ncdf(-d2) - spot * (-q * t).exp() * ncdf(-d1)
}

fn ncdf(x: f32) -> f32 {
    Normal::new(0.0, 1.0).unwrap().cdf(x as f64) as f32
}

fn npdf(x: f32) -> f32 {
    Normal::new(0.0, 1.0).unwrap().pdf(x as f64) as f32
}

#[derive(Debug)]
struct ContractResponse {
    is_call: bool,
    expiration_date: NaiveDate,
    strike: f32,
    last: f32,
    bid: f32,
    ask: f32,
    open_interest: u32,
    volatility: f32,
    delta: f32,
    gamma: f32,
    theta: f32,
    vega: f32,
    rho: f32,
}

impl ContractResponse {
    fn from(contract: &Contract, spot: f32, sim_date: NaiveDate) -> Self {
        // todo(chad): take risk-free rate and dividend into account
        let q = 0.0;
        // let r = 0.0;

        let expiration_date = NaiveDate::from_num_days_from_ce(contract.expiration_date);
        let dte = contract.expiration_date - sim_date.num_days_from_ce();
        let t = dte as f32 / 365.0; // todo(chad): 365 or 252??

        let volatility = contract.implied_volatility(spot, t);
        let sigma = volatility;

        let (d1, d2) = d(sigma, spot, contract.strike, t);
        let delta = (-q * t).exp() * ncdf(d1);
        let delta = if contract.is_call { delta } else { 1.0 - delta };

        let gamma = npdf(d1) / (spot * sigma.sqrt() * t.sqrt());

        let prob_density = (-d1 * d1 / 2.0).exp() / (2.0 as f32 * 3.14159 as f32).sqrt();
        let theta = -spot * prob_density * sigma / (2.0 * t.sqrt()) / 365.0 * 100.0;

        let vega = spot * prob_density * t.sqrt() / 100.0;

        let rho = if contract.is_call {
            contract.strike * t * ncdf(d2) / 100.0
        } else {
            -contract.strike * t * ncdf(-d2) / 100.0
        };

        Self {
            is_call: contract.is_call,
            expiration_date,
            strike: contract.strike,
            last: contract.last,
            bid: contract.bid,
            ask: contract.ask,
            open_interest: contract.open_interest,
            volatility,
            delta,
            gamma,
            theta,
            vega,
            rho,
        }
    }
}
