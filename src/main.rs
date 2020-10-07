extern crate actix_cors;
extern crate actix_web;
extern crate bincode;
extern crate chrono;
extern crate rand;
extern crate serde_derive;
extern crate serde_json;
extern crate statrs;

use std::collections::HashMap;
use std::io::Read;
use std::str;

use actix_cors::Cors;
use actix_web::{get, post, web, App, HttpRequest, HttpResponse, HttpServer, Responder};

use chrono::{Datelike, NaiveDate};

use serde::{Deserialize, Serialize};

use statrs::distribution::{Continuous, Normal, Univariate};

#[actix_rt::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .wrap(Cors::new().finish())
            .service(option_chain_for_symbol_and_date)
            .service(find_delta)
            .service(backtest)
    })
    //.bind("0.0.0.1:8000")?
    .bind("127.0.0.1:8000")?
    .run()
    .await
}

#[derive(Debug, Serialize, Deserialize)]
struct BacktestRequest {
    underlying: String,
    legs: Vec<BacktestRequestLeg>,

    take_profit_perc: Option<f32>,
    take_profit_dollars: Option<f32>,

    stop_loss_perc: Option<f32>,
    stop_loss_dollars: Option<f32>,

    stop_at_day: Option<i32>,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
enum ContractType {
    Call,
    Put,
    Equity,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
struct BacktestRequestLeg {
    contracts: i32,
    contract_type: ContractType,
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
    start_date: String,
    expiration_date: String,
    underlying_start_price: Option<f32>,
    strikes: Vec<Option<f32>>,
    pl: Vec<f32>,
}

enum ContractOrUnderlying {
    Contract(ContractResponse),
    Underlying(f32),
}

#[post("/backtest")]
async fn backtest(req: web::Json<BacktestRequest>) -> impl Responder {
    let mut response = BacktestResponse::default();

    let mut expiration_dates: Vec<String> = Vec::new();

    for year in 2005..=2020 {
        // for year in 2020..=2020 {
        let decoded = read_db_file(&year.to_string(), &req.underlying);
        if decoded.is_none() {
            continue;
        }

        let decoded = decoded.unwrap();
        let mut day_to_check = NaiveDate::from_ymd(year, 1, 1).num_days_from_ce();
        while !decoded.contains_key(&day_to_check) {
            day_to_check += 1;
        }

        let mut loop_limit = 0;
        loop {
            if !decoded.contains_key(&day_to_check) {
                break;
            }

            loop_limit += 1;
            if loop_limit > 365 {
                break;
            }

            let mut ed = Vec::new();
            for c in decoded[&day_to_check].contracts.iter() {
                let exp_date = NaiveDate::from_num_days_from_ce(c.expiration_date);
                let exp_date = exp_date.to_string();

                if !ed.contains(&exp_date) {
                    ed.push(exp_date);
                }
            }

            if ed.len() < 2 {
                continue;
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

    let mut db_cache = DbCache::default();

    let max_dte = req.legs.iter().map(|l| l.dte).max().unwrap();
    for expiration_date in expiration_dates.iter() {
        let expiration_date = NaiveDate::parse_from_str(expiration_date, "%Y-%m-%d").unwrap();
        let start_date =
            NaiveDate::from_num_days_from_ce(expiration_date.num_days_from_ce() - max_dte);

        // println!(
        //     "Running simulation for start_date: {}, expiration_date: {}",
        //     start_date, expiration_date
        // );

        // println!("{} / {}", index, expiration_dates.len());

        let leg_strikes = req
            .legs
            .iter()
            .map(|leg| {
                db_cache
                    .entry_closest_or_before(start_date.num_days_from_ce(), &req.underlying)
                    .map(|db_entry| match leg.contract_type {
                        ContractType::Call => {
                            let strike = find_strike_by_delta(
                                &db_entry,
                                leg.delta,
                                start_date,
                                expiration_date,
                                true,
                            );
                            strike
                        }
                        ContractType::Put => {
                            let strike = find_strike_by_delta(
                                &db_entry,
                                leg.delta,
                                start_date,
                                expiration_date,
                                false,
                            );
                            strike
                        }
                        ContractType::Equity => Some(db_entry.underlying_price / 100.0),
                    })
            })
            .collect::<Vec<_>>();

        if leg_strikes.iter().any(|ls| ls.is_none()) {
            continue;
        }

        let leg_strikes = leg_strikes.iter().map(|ls| ls.unwrap()).collect::<Vec<_>>();

        let mut trial = BacktestTrialResponse {
            start_date: start_date.to_string(),
            expiration_date: expiration_date.to_string(),
            underlying_start_price: db_cache
                .entry(start_date.num_days_from_ce(), &req.underlying)
                .map(|db_entry| db_entry.underlying_price),
            strikes: leg_strikes.clone(),
            pl: Vec::new(),
        };

        let mut premium_collected = None;
        let mut stopped_out = None;

        // if start_date == NaiveDate::from_ymd(2020, 6, 19) {
        //     let _stopme = 3;
        // }

        // TODO(chad): take different dte per leg into account
        for date_days in start_date.num_days_from_ce()..expiration_date.num_days_from_ce() {
            let date = NaiveDate::from_num_days_from_ce(date_days);

            let mut cost_day = 0.0;
            let mut cost_day_failed = false;

            for (leg_index, leg) in req.legs.iter().enumerate() {
                match leg_strikes[leg_index] {
                    Some(strike) => {
                        let contract = db_cache
                            .entry(date.num_days_from_ce(), &req.underlying)
                            .map(|db_entry| match leg.contract_type {
                                ContractType::Call => find_call_contract_by_strike(
                                    &db_entry,
                                    strike,
                                    date,
                                    expiration_date,
                                )
                                .map(|e| ContractOrUnderlying::Contract(e)),
                                ContractType::Put => find_put_contract_by_strike(
                                    &db_entry,
                                    strike,
                                    date,
                                    expiration_date,
                                )
                                .map(|e| ContractOrUnderlying::Contract(e)),
                                ContractType::Equity => Some(ContractOrUnderlying::Underlying(
                                    db_entry.underlying_price / 100.0,
                                )),
                            });

                        if let Some(Some(contract)) = contract {
                            if leg.is_buy {
                                match contract {
                                    ContractOrUnderlying::Contract(contract) => {
                                        cost_day -= contract.mid() * leg.contracts as f32;
                                    }
                                    ContractOrUnderlying::Underlying(price) => {
                                        cost_day -= price * leg.contracts as f32;
                                    }
                                }
                            } else {
                                match contract {
                                    ContractOrUnderlying::Contract(contract) => {
                                        cost_day += contract.mid() * leg.contracts as f32;
                                    }
                                    ContractOrUnderlying::Underlying(price) => {
                                        cost_day += price * leg.contracts as f32;
                                    }
                                }
                            }
                        } else {
                            cost_day_failed = true;
                        }
                    }
                    None => (),
                }
            }

            if !cost_day_failed {
                // if we have already stopped our loss or taken profit, simply use that
                // TODO(chad): can probably skip a lot of things if we have already taken profits or stopped a loss
                if let Some(stopped_out) = stopped_out {
                    trial.pl.push(stopped_out);
                    continue;
                }

                if let None = premium_collected {
                    premium_collected = Some(cost_day);
                }

                let profit_on_day = premium_collected.map(|p| p - cost_day);

                // Should we stop due to the day?
                let days_until_expiration = expiration_date.num_days_from_ce() - date_days;
                match (req.stop_at_day, stopped_out) {
                    (Some(sad), None) if days_until_expiration <= sad => {
                        stopped_out = Some(cost_day);
                    }
                    _ => (),
                }

                // Take profits due to percentage?
                match (premium_collected, req.take_profit_perc, profit_on_day) {
                    (Some(premium_collected), Some(rtp), Some(profit_on_day))
                        if profit_on_day > 0.0
                            && profit_on_day.abs() >= (rtp * premium_collected).abs() =>
                    {
                        stopped_out = Some(cost_day);
                    }
                    _ => (),
                }

                // Take profits due to dollars?
                match (req.take_profit_dollars, profit_on_day) {
                    (Some(rtp), Some(profit_on_day))
                        if profit_on_day > 0.0 && profit_on_day >= rtp =>
                    {
                        stopped_out = Some(cost_day);
                    }
                    _ => (),
                }

                // Stop loss due to percentage?
                match (premium_collected, req.stop_loss_perc, profit_on_day) {
                    (Some(premium_collected), Some(rsl), Some(profit_on_day))
                        if profit_on_day < 0.0
                            && profit_on_day.abs() >= (rsl * premium_collected).abs() =>
                    {
                        stopped_out = Some(cost_day);
                    }
                    _ => (),
                }

                // Stop loss due to percentage?
                match (req.stop_loss_dollars, profit_on_day) {
                    (Some(rsl), Some(profit_on_day))
                        if profit_on_day < 0.0 && profit_on_day.abs() >= rsl.abs() =>
                    {
                        stopped_out = Some(cost_day);
                    }
                    _ => (),
                }

                if stopped_out.is_some() {
                    trial.pl.push(stopped_out.unwrap());
                } else if stopped_out.is_none() {
                    trial.pl.push(cost_day);
                }
            } else if let Some(&last) = trial.pl.last() {
                trial.pl.push(last);
            }
        }

        if trial.strikes.iter().all(|s| s.is_some()) && !trial.pl.is_empty() {
            response.trials.push(trial);
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

    let call_strike = find_strike_by_delta(&decoded, delta, sim_date, exp_date, true);
    let put_strike = find_strike_by_delta(&decoded, delta, sim_date, exp_date, false);

    format!(
        "The price of {} on {} for the {} expiration cycle was: {}. The {} delta call option's strike price is: {:?}, and the {} delta put option's strike price is: {:?}\n",
        symbol, sim_date, exp_date, decoded.underlying_price, delta, call_strike, delta, put_strike,
    )
}

fn find_strike_by_delta(
    entry: &DbEntry,
    delta: f32,
    sim_date: NaiveDate,
    exp_date: NaiveDate,
    is_call: bool,
) -> Option<f32> {
    entry
        .contracts
        .iter()
        .filter(|contract| {
            let cr = ContractResponse::from(contract, entry.underlying_price, sim_date);
            cr.is_call == is_call
                && cr.delta <= (delta + 0.04)
                && cr.delta >= (delta - 0.04)
                && cr.expiration_date == exp_date
        })
        .map(|cr| cr.strike)
        .next()
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

    let exp_date = req.match_info().get("exp_date").unwrap();
    let exp_date = NaiveDate::parse_from_str(exp_date, "%Y-%m-%d").unwrap();

    let sim_date = req.match_info().get("sim_date").unwrap();
    let sim_date = NaiveDate::parse_from_str(sim_date, "%Y-%m-%d").unwrap();

    let strike = req.match_info().get("strike").unwrap();

    let decoded = read_db_file(&sim_date.year().to_string(), symbol);
    if decoded.is_none() {
        return format!("Failed to read db file");
    }
    let decoded = decoded.unwrap();
    let decoded = &decoded[&sim_date.num_days_from_ce()];

    let contracts = decoded
        .contracts
        .iter()
        .filter(|c| {
            c.strike == strike.parse::<f32>().unwrap()
                && c.expiration_date == exp_date.num_days_from_ce()
        })
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
    fn implied_volatility(&self, spot: f32, t: f32) -> f32 {
        let mut closest_vol = 0.0;

        let mut min_eps = std::f32::INFINITY;
        let threshold_eps = 0.005;

        let mut vol = 0.0;
        while vol < 3.0 {
            vol += 0.01;

            let theo_price = if self.is_call {
                call_price(vol, spot, self.strike, t)
            } else {
                put_price(vol, spot, self.strike, t)
            };

            let eps = (theo_price - self.mid()).abs();
            if eps < min_eps {
                min_eps = eps;
                closest_vol = vol;
            }

            if eps < threshold_eps {
                return vol;
            }
        }

        closest_vol
    }

    fn mid(&self) -> f32 {
        (self.bid + self.ask) / 2.0
    }
}

fn d(sigma: f32, spot: f32, strike: f32, t: f32) -> (f32, f32) {
    // TODO(chad): incorporate risk-free rate and dividend yield
    let r = 0.0;
    let q = 0.0;

    let d1 = ((spot / strike).ln() + t * (r - q + sigma * sigma / 2.0)) / (sigma * t.sqrt());
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

    fn mid(&self) -> f32 {
        (self.bid + self.ask) / 2.0
    }
}
