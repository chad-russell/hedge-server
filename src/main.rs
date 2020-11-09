extern crate actix_cors;
extern crate actix_web;
extern crate bincode;
extern crate chashmap;
extern crate chrono;
extern crate rand;
extern crate serde_derive;
extern crate serde_json;
extern crate statrs;

use std::collections::HashMap;
use std::io::Read;
use std::str;

use actix_cors::Cors;
use actix_web::{client, get, post, web, App, HttpRequest, HttpResponse, HttpServer, Responder};

use chashmap::CHashMap;

use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime};

use serde::{Deserialize, Serialize};

use rand::distributions::Distribution;
use statrs::distribution::{Continuous, Normal, Univariate};
use statrs::statistics::Variance;

const STD_DEV_LIMIT: f64 = 3.5;

#[actix_rt::main]
async fn main() -> std::io::Result<()> {
    let chain_cache = CHashMap::<String, ChainResponse>::new();
    let chain_cache_data = web::Data::new(chain_cache);

    HttpServer::new(move || {
        App::new()
            .wrap(Cors::new().finish())
            .app_data(chain_cache_data.clone())
            .service(option_chain_for_symbol_and_date)
            .service(get_chain)
            .service(find_delta)
            .service(backtest)
            .service(analyze)
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

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
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
    delta: Option<f32>,
    dte: Option<i32>,
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

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct AnalysisRequest {
    current_symbol: String,
    current_price: f64,
    open_positions: Vec<ContractPositionModel>,
    existing_pl: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ContractPositionModel {
    strike_price: Option<f64>,
    contract_type: ContractType,
    expiration_date: Option<String>,
    contracts: i64,
    cost: f64,
}

impl ContractPositionModel {
    fn is_expired(&self) -> bool {
        if self.contract_type == ContractType::Equity {
            return false;
        }

        if self.expiration_date.is_none() {
            return false;
        }

        let expiration_date =
            NaiveDate::parse_from_str(self.expiration_date.as_ref().unwrap(), "%m/%d/%Y");

        match expiration_date {
            Ok(expiration_date) => {
                expiration_date.and_time(NaiveTime::from_hms(0, 0, 0))
                    < chrono::offset::Utc::now().naive_utc()
            }
            _ => true,
        }
    }
}

#[derive(Debug, Serialize, Clone, Deserialize)]
struct OptionModel {
    #[serde(deserialize_with = "tda_f64_deser")]
    bid: Option<f64>,
    #[serde(deserialize_with = "tda_f64_deser")]
    ask: Option<f64>,
    #[serde(deserialize_with = "tda_f64_deser")]
    last: Option<f64>,
    #[serde(deserialize_with = "tda_f64_deser")]
    volatility: Option<f64>,
    #[serde(deserialize_with = "tda_f64_deser")]
    delta: Option<f64>,
    #[serde(deserialize_with = "tda_f64_deser")]
    gamma: Option<f64>,
    #[serde(deserialize_with = "tda_f64_deser")]
    theta: Option<f64>,
    #[serde(deserialize_with = "tda_f64_deser")]
    vega: Option<f64>,
}

fn tda_f64_deser<'de, D>(de: D) -> Result<Option<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let deser_res: Result<serde_json::Value, D::Error> = serde::Deserialize::deserialize(de);

    match deser_res {
        Ok(serde_json::Value::Number(num)) => Ok(num.as_f64()),
        Ok(serde_json::Value::String(s)) => {
            let f: Result<f64, _> = s.parse();
            match f {
                Ok(f) => Ok(Some(f)),
                _ => Ok(None),
            }
        }
        _ => Ok(None),
    }
}

enum ContractOrUnderlying {
    Contract(ContractResponse),
    Underlying(f32),
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct IVolatilityRecord {
    symbol: String,
    stock_id: Option<i64>,
    expiration_date: String,
    strike: f64,
    #[serde(rename = "type")]
    type_: String,
    style: Option<String>,
    open_interest: Option<i64>,
    bid_price: f64,
    ask_price: f64,
    last_price: Option<f64>,
    bid_date: Option<i64>,
    ask_date: Option<i64>,
    last_date: Option<i64>,
    bid_size: Option<i64>,
    ask_size: Option<i64>,
    last_size: Option<i64>,
    bid_exchange: Option<String>,
    ask_exchange: Option<String>,
    last_exchange: Option<String>,
    volume: Option<i64>,
    cumulative_volume: Option<i64>,
    timestamp: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PolygonResponse {
    last: PolygonLast,
}

#[derive(Debug, Deserialize)]
struct PolygonLast {
    price: f32,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct TDAmeritradeOptionsResponse {
    symbol: String,
    underlying: TDAmeritradeUnderlying,
    put_exp_date_map: HashMap<String, HashMap<String, Vec<OptionModel>>>,
    call_exp_date_map: HashMap<String, HashMap<String, Vec<OptionModel>>>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct TDAmeritradeUnderlying {
    last: Option<f64>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ChainResponse {
    symbol: String,
    put_exp_date_map: HashMap<String, HashMap<String, OptionModel>>,
    call_exp_date_map: HashMap<String, HashMap<String, OptionModel>>,
    #[serde(skip_serializing)]
    timestamp: NaiveDateTime,
}

#[get("/chain/{symbol}")]
async fn get_chain(
    req: HttpRequest,
    data: web::Data<CHashMap<String, ChainResponse>>,
) -> impl Responder {
    let symbol = req.match_info().get("symbol").unwrap();
    HttpResponse::Ok().json(get_chain_impl(symbol, &data).await)
}

async fn last_price_for(client: &client::Client, symbol: &str) -> f32 {
    let mut i: i32 = 0;
    let mut is_err = true;

    while i < 5 && is_err {
        let polygon_response = client.get(&format!("https://api.polygon.io/v1/last/stocks/{}?apiKey=j_dJPmbEXvS7y_QHQOglcY_NnnjbUIad2l89Ie", symbol))
        .send()
        .await;
        let body = polygon_response.unwrap().body().await.unwrap();

        let polygon_response: Result<PolygonResponse, serde_json::Error> =
            serde_json::from_reader(&*body);

        if polygon_response.is_ok() {
            let spot_price = polygon_response.unwrap().last.price;
            return spot_price;
        } else {
            println!("{:?}", body)
        }

        is_err = polygon_response.is_err();
        i += 1;
    }

    panic!();
}

async fn get_tda_chain(
    client: &client::Client,
    symbol: &str,
) -> Result<
    (
        HashMap<String, HashMap<String, OptionModel>>,
        HashMap<String, HashMap<String, OptionModel>>,
    ),
    (),
> {
    let body = client.get(
          &format!("https://api.tdameritrade.com/v1/marketdata/chains?apikey=EXURQJNFBLLVGWUXKZ7N06MWWBKKRKWD&symbol={}&includeQuotes=TRUE", symbol))
          .timeout(core::time::Duration::from_secs(90))
          .send()
          .await
          .map_err(|_| ())?
          .body()
          .limit(1_000_000_000_000)
          .await
          .map_err(|_| ())?;

    let response: TDAmeritradeOptionsResponse = serde_json::from_reader(&*body).map_err(|_| ())?;
    let put_exp_date_map = response
        .put_exp_date_map
        .iter()
        .map(|(key, value)| {
            let colon_index = key.find(':').unwrap();
            let parsed_date = NaiveDate::parse_from_str(&key[0..colon_index], "%Y-%m-%d").unwrap();
            let formatted_date = parsed_date.format("%Y-%m-%d").to_string();
            let value = value
                .iter()
                .map(|(key, value)| {
                    let mut value = value[0].clone();
                    value.volatility = value.volatility.map(|v| v / 100.0);

                    let compute_volatility = match value.volatility {
                        None => true,
                        Some(f) if f.is_nan() => true,
                        _ => false,
                    };
                    if compute_volatility {
                        let dte = parsed_date - chrono::offset::Utc::now().naive_utc().date();
                        let dte = dte.num_days() as usize;
                        let t = dte as f32 / 365.0; // todo(chad): 365 or 252??

                        let volatility = implied_volatility(
                            response.underlying.last.unwrap() as _,
                            key.parse::<f32>().unwrap(),
                            false,
                            value.last.unwrap() as _,
                            t,
                        );

                        value.volatility = Some(volatility as f64);
                    }

                    (key.clone(), value)
                })
                .collect();

            (formatted_date, value)
        })
        .collect();

    let call_exp_date_map = response
        .call_exp_date_map
        .iter()
        .map(|(key, value)| {
            let colon_index = key.find(':').unwrap();
            let parsed_date = NaiveDate::parse_from_str(&key[0..colon_index], "%Y-%m-%d").unwrap();
            let formatted_date = parsed_date.format("%Y-%m-%d").to_string();
            let value = value
                .iter()
                .map(|(key, value)| {
                    let mut value = value[0].clone();
                    value.volatility = value.volatility.map(|v| v / 100.0);

                    let compute_volatility = match value.volatility {
                        None => true,
                        Some(f) if f.is_nan() => true,
                        _ => false,
                    };
                    if compute_volatility {
                        let dte = parsed_date - chrono::offset::Utc::now().naive_utc().date();
                        let dte = dte.num_days() as usize;
                        let t = dte as f32 / 365.0; // todo(chad): 365 or 252??

                        let volatility = implied_volatility(
                            response.underlying.last.unwrap() as _,
                            key.parse::<f32>().unwrap(),
                            true,
                            value.last.unwrap() as _,
                            t,
                        );

                        value.volatility = Some(volatility as f64);
                    }

                    (key.clone(), value)
                })
                .collect();

            (formatted_date, value)
        })
        .collect();

    Ok((call_exp_date_map, put_exp_date_map))
}

async fn get_ivol_chain(
    spot_price: f32,
    call_exp_date_map: &mut HashMap<String, HashMap<String, OptionModel>>,
    put_exp_date_map: &mut HashMap<String, HashMap<String, OptionModel>>,
    rdr: &mut csv::Reader<&[u8]>,
) -> i32 {
    let mut call_exp_date_map = call_exp_date_map;
    let mut put_exp_date_map = put_exp_date_map;

    let mut record_count = 0;
    for record in rdr.records() {
        record_count += 1;
        let record = match record {
            Ok(record) => record,
            Err(e) => {
                println!("Bad Record! (1) {:#?}", e);
                continue;
            }
        };
        let record: Result<IVolatilityRecord, csv::Error> = record.deserialize(None);

        let record = match record {
            Ok(record) => record,
            Err(e) => {
                println!("Bad Record! (2) {:#?}", e);
                continue;
            }
        };

        let date = record.expiration_date;
        let parsed_date = NaiveDate::parse_from_str(&date, "%Y-%m-%dT%H:%M:%S.%f%z").unwrap();
        let strike_string = serde_json::Value::from(record.strike).to_string();

        let map = if record.type_ == "C" {
            &mut call_exp_date_map
        } else if record.type_ == "P" {
            &mut put_exp_date_map
        } else {
            println!("Invalid type {}, should be 'C' or 'P'", record.type_);
            continue;
        };

        let entry = map
            .entry(parsed_date.format("%Y-%m-%d").to_string())
            .or_insert(HashMap::new());

        // TODO(chad)
        let today = chrono::offset::Local::today();
        let t = (parsed_date.num_days_from_ce() as f32 - today.num_days_from_ce() as f32) / 365.0;

        let volatility = implied_volatility(
            spot_price,
            record.strike as _,
            record.type_ == "C",
            record
                .last_price
                .unwrap_or((record.bid_price + record.ask_price) / 2.0) as _,
            t,
        ) as f64;

        entry.insert(
            strike_string,
            OptionModel {
                bid: Some(record.bid_price),
                ask: Some(record.ask_price),
                last: record.last_price,
                volatility: Some(volatility),
                delta: if record.type_ == "C" {
                    Some(-call_delta(
                        spot_price as _,
                        record.strike,
                        volatility,
                        t as _,
                    ))
                } else {
                    Some(-put_delta(
                        spot_price as _,
                        record.strike,
                        volatility,
                        t as _,
                    ))
                },
                theta: Some(call_theta(spot_price as _, record.strike, volatility, t as _) / 100.0),
                gamma: Some(0.0),
                vega: Some(0.0),
            },
        );
    }

    record_count
}

async fn get_chain_impl(
    symbol: &str,
    cache: &web::Data<CHashMap<String, ChainResponse>>,
) -> ChainResponse {
    // Try to look up the result in the cache
    match cache.get(symbol) {
        Some(entry) => {
            let duration = chrono::offset::Utc::now().naive_utc() - entry.timestamp;
            if duration < chrono::Duration::minutes(5) {
                return entry.clone();
            }
        }
        None => {}
    }

    let client = client::Client::default();

    let spot_price = last_price_for(&client, symbol).await;

    // HashMap<i64, HashMap<F64String, OptionModel>>
    let mut call_exp_date_map = HashMap::new();
    let mut put_exp_date_map = HashMap::new();

    let tda = get_tda_chain(&client, symbol).await;
    match tda {
        Ok((c, p)) => {
            call_exp_date_map = c;
            put_exp_date_map = p;
        }
        _ => {
            let mut ivol_res = client.get(&format!("https://restapi.ivolatility.com/quotes/options?symbol={}&username=dougrussel&password=UsEVO0Ei", symbol))
        .timeout(core::time::Duration::from_secs(90))
        .send()
        .await
        .unwrap();

            let status = ivol_res.status();
            if status.is_success() {
                let body = ivol_res.body().limit(1_000_000_000_000).await.unwrap();
                let mut rdr = csv::Reader::from_reader(&*body);
                let _record_count = get_ivol_chain(
                    spot_price,
                    &mut call_exp_date_map,
                    &mut put_exp_date_map,
                    &mut rdr,
                )
                .await;
                // println!("For {}: got {} records", symbol, record_count);
                // if record_count < 100 {
                //     println!("body: {}", str::from_utf8(&*body).unwrap());
                // }
            }
        }
    }

    let response = ChainResponse {
        symbol: symbol.to_string(),
        put_exp_date_map,
        call_exp_date_map,
        timestamp: chrono::offset::Utc::now().naive_utc(),
    };

    cache.insert(symbol.into(), response.clone());

    response
}

#[derive(Debug, Serialize, Deserialize)]
struct AnalysisModelResponse {
    message: AnalysisRequest,

    simulated_days_length: i64,

    max_gain: f64,
    max_loss: f64,

    min_delta: f64,
    max_delta: f64,

    min_theta: f64,
    max_theta: f64,

    min_low: f64,
    max_high: f64,

    volatility_estimate: f64,

    // 100 doubles; cumulative distribution
    daily_distributions: Vec<Vec<f64>>,
    daily_stats: Vec<f64>, // standard deviations

    pop: f64,
    p50: f64,
}

impl AnalysisModelResponse {
    fn new(message: AnalysisRequest) -> Self {
        Self {
            message,
            simulated_days_length: 0,
            max_gain: 0.0,
            max_loss: 0.0,
            min_delta: 0.0,
            max_delta: 0.0,
            min_theta: 0.0,
            max_theta: 0.0,
            min_low: 0.0,
            max_high: 0.0,
            volatility_estimate: 0.0,
            daily_distributions: Vec::new(),
            daily_stats: Vec::new(),
            pop: 0.0,
            p50: 0.0,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
struct SimulatedDay {
    underlying_price: f64,
    position_pl: f64,
}

impl SimulatedDay {
    fn new(underlying_price: f64, position_pl: f64) -> Self {
        Self {
            underlying_price,
            position_pl,
        }
    }
}

#[post("/analyze")]
async fn analyze(
    req: web::Json<AnalysisRequest>,
    data: web::Data<CHashMap<String, ChainResponse>>,
) -> impl Responder {
    let mut rng = rand::thread_rng();

    let mut response = AnalysisModelResponse::new(req.clone());

    let message = &response.message;
    if message.open_positions.is_empty() {
        return HttpResponse::Ok().json(response);
    }

    let mut chain = get_chain_impl(&message.current_symbol, &data).await;

    let last_expiration_date: Vec<NaiveDate> = message
        .open_positions
        .iter()
        .filter(|p| p.expiration_date.is_some())
        .map(|p| NaiveDate::parse_from_str(p.expiration_date.as_ref().unwrap(), "%m/%d/%Y"))
        .filter(|e| e.is_ok())
        .map(|e| e.unwrap())
        .collect::<Vec<_>>();
    let last_expiration_date = last_expiration_date.iter().max();
    if last_expiration_date.is_none() {
        return HttpResponse::Ok().json(response);
    }
    let last_expiration_date = *last_expiration_date.unwrap();

    let open_contracts = message
        .open_positions
        .iter()
        .map(|p| {
            if p.contract_type == ContractType::Equity {
                return None;
            }
            if p.is_expired() {
                return None;
            }

            let contract_map = if p.contract_type == ContractType::Call {
                &chain.call_exp_date_map
            } else {
                &chain.put_exp_date_map
            };

            let ed =
                NaiveDate::parse_from_str(p.expiration_date.as_ref().unwrap(), "%m/%d/%Y").unwrap();
            let formatted_ed = ed.format("%Y-%m-%d").to_string();

            let contract_date_map = match contract_map.get(&formatted_ed) {
                Some(c) => c,
                None => {
                    return None;
                }
            };

            let found = contract_date_map
                .get(&(serde_json::Value::from(p.strike_price.unwrap()).to_string()));

            found.cloned()
        })
        .filter(|p| p.is_some())
        .map(|e| e.unwrap())
        .collect::<Vec<_>>();

    if open_contracts.is_empty() {
        return HttpResponse::Ok().json(response);
    }

    let volatilities = open_contracts
        .iter()
        .map(|c| c.volatility)
        .collect::<Vec<_>>();

    response.volatility_estimate = volatilities
        .iter()
        .filter(|v| v.is_some())
        .map(|v| v.unwrap())
        .sum::<f64>()
        / volatilities.len() as f64;

    // println!("volatility estimate: {}", response.volatility_estimate);
    // position_price(req.current_price, 0.0, &req, &chain, true);

    let days_until_expiration =
        last_expiration_date - chrono::offset::Utc::now().naive_utc().date();
    let days_until_expiration = days_until_expiration.num_days() as usize;
    response.simulated_days_length = days_until_expiration as _;

    let mut simulated_days = vec![Vec::<SimulatedDay>::new(); days_until_expiration];

    for _ in 0..1500 {
        for i in 0..days_until_expiration {
            if i == 0 {
                let daily_volatility = response.volatility_estimate * (i as f64 / 365.0).sqrt();
                let daily_std_dev = daily_volatility * message.current_price;

                let mut price = message.current_price;

                let sample = if i == 0 {
                    0.0
                } else {
                    Normal::new(1.0, daily_std_dev).unwrap().sample(&mut rng)
                };
                price += sample;
                if price < 0.01 {
                    price = 0.01;
                }

                simulated_days[i].push(SimulatedDay::new(
                    price,
                    position_price(price, i as _, message, &mut chain, false),
                ));
            } else {
                let prev_day = simulated_days[i - 1].last().unwrap();

                let daily_volatility = response.volatility_estimate * (1.0 / 365.0 as f64).sqrt();
                let daily_std_dev = daily_volatility * message.current_price;

                let mut price = prev_day.underlying_price;
                let sample = Normal::new(0.0, daily_std_dev).unwrap().sample(&mut rng);

                let daily_expected_move = response.volatility_estimate
                    * message.current_price
                    * (i as f64 / 365.0).sqrt()
                    * STD_DEV_LIMIT;
                let equity_price = message.current_price;

                price += sample;
                if price > equity_price + daily_expected_move {
                    price = equity_price + daily_expected_move;
                }
                if price < equity_price - daily_expected_move {
                    price = equity_price - daily_expected_move;
                }
                if price < 0.01 {
                    price = 0.01;
                }

                simulated_days[i].push(SimulatedDay::new(
                    price,
                    position_price(price, i as _, message, &mut chain, false),
                ));
            }
        }
    }

    if simulated_days.is_empty() {
        return HttpResponse::Ok().json(response);
    }

    response.volatility_estimate *= message.current_price;

    let estimated_max_move =
        (simulated_days.len() as f64 / 365.0).sqrt() * response.volatility_estimate * STD_DEV_LIMIT;

    response.max_high = message.current_price + estimated_max_move;
    response.min_low = (message.current_price - estimated_max_move).max(0.01);

    response.max_gain = std::f64::NEG_INFINITY;
    response.max_loss = std::f64::INFINITY;

    response.max_delta = std::f64::NEG_INFINITY;
    response.min_delta = std::f64::INFINITY;

    response.max_theta = std::f64::NEG_INFINITY;
    response.min_theta = std::f64::INFINITY;

    let mut sim_day = 0.0;
    while sim_day < simulated_days.len() as _ {
        for i in 1..500 {
            let price =
                response.min_low + (response.max_high - response.min_low) * i as f64 / 500.0;

            let (pl, delta, theta) =
                position_price_delta_theta(price, sim_day as f64, message, &chain);

            // let pl = position_price(price, sim_day as f64, message, &chain);
            if pl > response.max_gain {
                response.max_gain = pl;
            }
            if pl < response.max_loss {
                response.max_loss = pl;
            }

            // let delta = position_delta(price, sim_day, message, &chain);
            if delta > response.max_delta {
                response.max_delta = delta;
            }
            if delta < response.min_delta {
                response.min_delta = delta;
            }

            // let theta = position_theta(price, sim_day, message, &chain);
            if theta > response.max_theta {
                response.max_theta = theta;
            }
            if theta < response.min_theta {
                response.min_theta = theta;
            }
        }

        sim_day += 0.1;
    }

    response.daily_distributions = Vec::with_capacity(simulated_days.len());
    for day in simulated_days.iter() {
        let low = response.max_loss;
        let high = response.max_gain;

        let mut cum = Vec::new();
        let mut i = low;
        while i < high {
            cum.push(
                day.iter()
                    .filter(|d| d.position_pl >= i)
                    .collect::<Vec<_>>()
                    .len() as f64
                    / day.len() as f64,
            );

            i += (high - low) / 100.0;
        }
        response.daily_distributions.push(cum);

        response.daily_stats.push(
            day.iter()
                .map(|e| e.position_pl)
                .collect::<Vec<_>>()
                .as_slice()
                .std_dev(),
        );
    }

    let mut daily_distribution_index_pop =
        (-response.max_loss / (response.max_gain - response.max_loss) * 100.0).round() as i64;
    if daily_distribution_index_pop < 0 {
        daily_distribution_index_pop = 0;
    }
    if daily_distribution_index_pop > 99 {
        daily_distribution_index_pop = 99;
    }

    let mut daily_distribution_index_p50 = ((response.max_gain / 2.0 - response.max_loss)
        / (response.max_gain - response.max_loss)
        * 100.0)
        .round() as i64;
    if daily_distribution_index_p50 < 0 {
        daily_distribution_index_p50 = 0;
    }
    if daily_distribution_index_p50 > 99 {
        daily_distribution_index_p50 = 99;
    }

    response.pop =
        response.daily_distributions.last().unwrap()[daily_distribution_index_pop as usize] * 100.0;
    response.p50 =
        response.daily_distributions.last().unwrap()[daily_distribution_index_p50 as usize] * 100.0;

    HttpResponse::Ok().json(response)
}

// fn position_delta(
//     price: f64,
//     days_in_future: f64,
//     message: &AnalysisRequest,
//     chain: &ChainResponse,
// ) -> f64 {
//     let mut answer = 0.0;

//     for p in message.open_positions.iter() {
//         if p.contract_type == ContractType::Equity {
//             answer += p.contracts as f64;
//             continue;
//         }

//         let parsed_date = NaiveDate::parse_from_str(&p.expiration_date, "%m/%d/%Y").unwrap();

//         let days_until_expiration = parsed_date - chrono::offset::Utc::now().naive_utc().date();
//         let days_until_expiration = days_until_expiration.num_days() as usize;

//         let t = (days_until_expiration as f64 - days_in_future) / 365.0;

//         // Skip all expired
//         if t < 0.0 {
//             continue;
//         }

//         let position_contract_map = if p.contract_type == ContractType::Call {
//             &chain.call_exp_date_map
//         } else {
//             &chain.put_exp_date_map
//         };

//         let parsed_date = NaiveDate::parse_from_str(&p.expiration_date, "%m/%d/%Y").unwrap();

//         let position_contract_by_date =
//             position_contract_map.get(&parsed_date.format("%Y-%m-%d").to_string());
//         if position_contract_by_date.is_none() {
//             continue;
//         }

//         let position_contract = position_contract_by_date
//             .unwrap()
//             .get(&p.strike_price.to_string());

//         let position_delta = if position_contract.is_none() || p.is_expired() || p.contracts == 0 {
//             p.cost
//         } else if p.contract_type == ContractType::Call {
//             p.contracts.abs() as f64
//                 * call_delta(
//                     position_contract.unwrap().volatility as f64,
//                     price as _,
//                     p.strike_price as _,
//                     t as _,
//                 )
//         } else {
//             p.contracts.abs() as f64
//                 * put_delta(
//                     position_contract.unwrap().volatility as f64,
//                     price as _,
//                     p.strike_price as _,
//                     t as _,
//                 )
//         };

//         answer += position_delta;
//     }

//     answer
// }

// fn position_theta(
//     price: f64,
//     days_in_future: f64,
//     message: &AnalysisRequest,
//     chain: &ChainResponse,
// ) -> f64 {
//     let mut answer = 0.0;

//     for (i, p) in message.open_positions.iter().enumerate() {
//         if p.contract_type == ContractType::Equity {
//             continue;
//         }

//         let parsed_date = NaiveDate::parse_from_str(&p.expiration_date, "%m/%d/%Y").unwrap();

//         let days_until_expiration = parsed_date - chrono::offset::Utc::now().naive_utc().date();
//         let days_until_expiration = days_until_expiration.num_days() as usize;

//         let t = (days_until_expiration as f64 - days_in_future) / 365.0;

//         // Skip all expired
//         if t < 0.0 {
//             continue;
//         }

//         let position_contract_map = if p.contract_type == ContractType::Call {
//             &chain.call_exp_date_map
//         } else {
//             &chain.put_exp_date_map
//         };

//         let parsed_date = NaiveDate::parse_from_str(&p.expiration_date, "%m/%d/%Y").unwrap();

//         let position_contract_by_date =
//             position_contract_map.get(&parsed_date.format("%Y-%m-%d").to_string());
//         if position_contract_by_date.is_none() {
//             continue;
//         }

//         let position_contract = position_contract_by_date
//             .unwrap()
//             .get(&p.strike_price.to_string());

//         let position_theta = if position_contract.is_none() || p.is_expired() || p.contracts == 0 {
//             0.0
//         } else if p.contract_type == ContractType::Call {
//             p.contracts.abs() as f64
//                 * call_theta(
//                     position_contract.unwrap().volatility as f64,
//                     price as _,
//                     p.strike_price as _,
//                     t as _,
//                 )
//         } else {
//             p.contracts.abs() as f64
//                 * put_theta(
//                     position_contract.unwrap().volatility as f64,
//                     price as _,
//                     p.strike_price as _,
//                     t as _,
//                 )
//         };

//         answer += position_theta;
//     }

//     answer
// }

fn position_price_delta_theta(
    price: f64,
    days_in_future: f64,
    message: &AnalysisRequest,
    chain: &ChainResponse,
) -> (f64, f64, f64) {
    let mut answer_price = message.existing_pl;
    let mut answer_delta = 0.0;
    let mut answer_theta = 0.0;

    for p in message.open_positions.iter() {
        if p.contract_type == ContractType::Equity {
            answer_price += price * p.contracts as f64 + p.cost;
            answer_delta += p.contracts as f64;
            continue;
        }

        let parsed_date =
            NaiveDate::parse_from_str(p.expiration_date.as_ref().unwrap(), "%m/%d/%Y").unwrap();

        let days_until_expiration = parsed_date - chrono::offset::Utc::now().naive_utc().date();
        let days_until_expiration = days_until_expiration.num_days() as usize;

        let mut t = (days_until_expiration as f64 - days_in_future) / 365.0;

        // If this is expired, just return the value at expiration
        // if (t < 0.1 / 365) {
        //   t = 0.1 / 365;
        // }
        if t < 0.0 {
            t = 0.0;
        }

        let position_contract_map = if p.contract_type == ContractType::Call {
            &chain.call_exp_date_map
        } else {
            &chain.put_exp_date_map
        };

        let parsed_date =
            NaiveDate::parse_from_str(p.expiration_date.as_ref().unwrap(), "%m/%d/%Y").unwrap();

        let position_contract_by_date =
            position_contract_map.get(&parsed_date.format("%Y-%m-%d").to_string());
        if position_contract_by_date.is_none() {
            continue;
        }

        let position_contract = position_contract_by_date
            .unwrap()
            .get(&(serde_json::Value::from(p.strike_price.unwrap()).to_string()));

        let contract_price = if position_contract.is_none() || p.is_expired() || p.contracts == 0 {
            if p.contract_type == ContractType::Put || p.contract_type == ContractType::Call {
                p.cost / 100.0
            } else {
                p.cost
            }
        } else if p.contract_type == ContractType::Call {
            p.contracts.abs() as f64
                * call_price_safe(
                    position_contract.unwrap().volatility,
                    price as _,
                    p.strike_price.unwrap() as _,
                    t as _,
                ) as f64
        } else {
            p.contracts.abs() as f64
                * put_price_safe(
                    position_contract.unwrap().volatility,
                    price as _,
                    p.strike_price.unwrap() as _,
                    t as _,
                ) as f64
        };

        let position_cost = p.cost.abs();

        // short options
        let mut pl = position_cost - contract_price * 100.0;

        // long options
        if p.contracts > 0 {
            pl = -pl;
        }

        answer_price += pl;

        if t > 0.0 {
            let position_delta =
                if position_contract.is_none() || p.is_expired() || p.contracts == 0 {
                    p.cost
                } else if p.contract_type == ContractType::Call {
                    p.contracts.abs() as f64
                        * call_delta_safe(
                            price as _,
                            p.strike_price.unwrap() as _,
                            position_contract.unwrap().volatility as _,
                            t as _,
                        )
                } else {
                    p.contracts.abs() as f64
                        * put_delta_safe(
                            price as _,
                            p.strike_price.unwrap() as _,
                            position_contract.unwrap().volatility as _,
                            t as _,
                        )
                };
            answer_delta += position_delta;
        }

        if t > 0.0 {
            let position_theta =
                if position_contract.is_none() || p.is_expired() || p.contracts == 0 {
                    p.cost
                } else if p.contract_type == ContractType::Call {
                    p.contracts.abs() as f64
                        * call_theta_safe(
                            price as _,
                            p.strike_price.unwrap() as _,
                            position_contract.unwrap().volatility as _,
                            t as _,
                        )
                } else {
                    p.contracts.abs() as f64
                        * call_theta_safe(
                            price as _,
                            p.strike_price.unwrap() as _,
                            position_contract.unwrap().volatility as _,
                            t as _,
                        )
                };
            answer_theta += position_theta;
        }
    }

    (answer_price, answer_delta, answer_theta)
}

fn position_price(
    price: f64,
    days_in_future: f64,
    message: &AnalysisRequest,
    chain: &ChainResponse,
    debug_contract_price: bool,
) -> f64 {
    let mut answer = message.existing_pl;

    for p in message.open_positions.iter() {
        if p.contract_type == ContractType::Equity {
            answer += price * p.contracts as f64 + p.cost;
            continue;
        }

        let parsed_date =
            NaiveDate::parse_from_str(p.expiration_date.as_ref().unwrap(), "%m/%d/%Y").unwrap();

        let days_until_expiration = parsed_date - chrono::offset::Utc::now().naive_utc().date();
        let days_until_expiration = days_until_expiration.num_days() as usize;

        let mut t = (days_until_expiration as f64 - days_in_future) / 365.0;

        // If this is expired, just return the value at expiration
        // if (t < 0.1 / 365) {
        //   t = 0.1 / 365;
        // }
        if t < 0.0 {
            t = 0.0;
        }

        let position_contract_map = if p.contract_type == ContractType::Call {
            &chain.call_exp_date_map
        } else {
            &chain.put_exp_date_map
        };

        let parsed_date =
            NaiveDate::parse_from_str(p.expiration_date.as_ref().unwrap(), "%m/%d/%Y").unwrap();

        let position_contract_by_date =
            position_contract_map.get(&parsed_date.format("%Y-%m-%d").to_string());
        if position_contract_by_date.is_none() {
            continue;
        }

        let position_contract = position_contract_by_date
            .unwrap()
            .get(&(serde_json::Value::from(p.strike_price.unwrap()).to_string()));

        let contract_price = if position_contract.is_none() || p.is_expired() || p.contracts == 0 {
            if p.contract_type == ContractType::Put || p.contract_type == ContractType::Call {
                p.cost / 100.0
            } else {
                p.cost
            }
        } else if p.contract_type == ContractType::Call {
            p.contracts.abs() as f64
                * call_price_safe(
                    position_contract.unwrap().volatility,
                    price as _,
                    p.strike_price.unwrap() as _,
                    t as _,
                ) as f64
        } else {
            p.contracts.abs() as f64
                * put_price_safe(
                    position_contract.unwrap().volatility,
                    price as _,
                    p.strike_price.unwrap() as _,
                    t as _,
                ) as f64
        };

        if debug_contract_price {
            println!("right now price: {}", contract_price);
        }

        let position_cost = p.cost.abs();

        // short options
        let mut pl = position_cost - contract_price * 100.0;

        // long options
        if p.contracts > 0 {
            pl = -pl;
        }

        answer += pl;
    }

    answer
}

fn call_delta(
    s0: f64,    // underlying price
    x: f64,     // strike price
    sigma: f64, // volatility
    t: f64,     // time to expiration (% of one year)
) -> f64 {
    let q = 0.0;

    let (d1, _) = d(sigma as _, s0 as _, x as _, t as _);
    -((-q * t).exp()) * ncdf(d1) as f64
}

fn call_delta_safe(
    s0: f64,            // underlying price
    x: f64,             // strike price
    sigma: Option<f64>, // volatility
    t: f64,             // time to expiration (% of one year)
) -> f64 {
    match sigma {
        Some(sigma) => call_delta(s0, x, sigma, t),
        _ => 0.0,
    }
}

fn put_delta(
    s0: f64,    // underlying price
    x: f64,     // strike price
    sigma: f64, // volatility
    t: f64,     // time to expiration (% of one year)
) -> f64 {
    let q = 0.0;

    let (d1, _) = d(sigma as _, s0 as _, x as _, t as _);
    -((-q * t).exp()) * (ncdf(d1) - 1.0) as f64
}

fn put_delta_safe(
    s0: f64,            // underlying price
    x: f64,             // strike price
    sigma: Option<f64>, // volatility
    t: f64,             // time to expiration (% of one year)
) -> f64 {
    match sigma {
        Some(sigma) => put_delta(s0, x, sigma, t),
        _ => 0.0,
    }
}

fn call_theta(
    s0: f64,    // underlying price
    x: f64,     // strike price
    sigma: f64, // volatility
    t: f64,     // time to expiration (% of one year)
) -> f64 {
    let (d1, _) = d(sigma as _, s0 as _, x as _, t as _);
    let prob_density = (-d1 * d1 / 2.0).exp() as f64 / (2.0 * 3.14159 as f64).sqrt();
    let theta = -s0 * prob_density * sigma / (2.0 * t.sqrt()) / 365.0 * 100.0;
    theta
}

fn call_theta_safe(
    s0: f64,            // underlying price
    x: f64,             // strike price
    sigma: Option<f64>, // volatility
    t: f64,             // time to expiration (% of one year)
) -> f64 {
    match sigma {
        Some(sigma) => call_theta(s0, x, sigma, t),
        _ => 0.0,
    }
}

#[post("/backtest")]
async fn backtest(req: web::Json<BacktestRequest>) -> impl Responder {
    let mut response = BacktestResponse::default();

    let mut expiration_dates: Vec<String> = Vec::new();

    for year in 2005..=2020 {
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

    let max_dte = req
        .legs
        .iter()
        .map(|l| l.dte)
        .filter(|dte| dte.is_some())
        .map(|dte| dte.unwrap())
        .max()
        .unwrap();
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
                                leg.delta.unwrap(),
                                start_date,
                                expiration_date,
                                true,
                            );
                            strike
                        }
                        ContractType::Put => {
                            let strike = find_strike_by_delta(
                                &db_entry,
                                leg.delta.unwrap(),
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
        implied_volatility(spot, self.strike, self.is_call, self.mid(), t)
    }

    fn mid(&self) -> f32 {
        (self.bid + self.ask) / 2.0
    }
}

fn implied_volatility(spot: f32, strike: f32, is_call: bool, price: f32, t: f32) -> f32 {
    let mut closest_vol = 0.0;

    let mut min_eps = std::f32::INFINITY;
    let threshold_eps = 0.005;

    let mut vol = 0.0;
    while vol < 3.0 {
        vol += 0.01;

        let theo_price = if is_call {
            call_price(vol, spot, strike, t)
        } else {
            put_price(vol, spot, strike, t)
        };

        let eps = (theo_price - price).abs();
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

fn call_price_safe(sigma: Option<f64>, spot: f32, strike: f32, t: f32) -> f32 {
    match sigma {
        Some(sigma) => call_price(sigma as _, spot, strike, t),
        _ => 0.0,
    }
}

fn put_price(sigma: f32, spot: f32, strike: f32, t: f32) -> f32 {
    // TODO(chad): incorporate risk-free rate and dividend yield
    let r = 0.0;
    let q = 0.0;

    let (d1, d2) = d(sigma, spot, strike, t);

    strike * (-r * t).exp() * ncdf(-d2) - spot * (-q * t).exp() * ncdf(-d1)
}

fn put_price_safe(sigma: Option<f64>, spot: f32, strike: f32, t: f32) -> f32 {
    match sigma {
        Some(sigma) => put_price(sigma as _, spot, strike, t),
        _ => 0.0,
    }
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
