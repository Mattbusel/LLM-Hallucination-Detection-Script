use warp::Filter;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use tokio::time::{Duration, interval};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardStats {
    pub total_analyses: usize,
    pub average_hallucination_rate: f32,
    pub analyses_per_minute: f32,
    pub top_issues: Vec<String>,
    pub confidence_distribution: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisRequest {
    pub text: String,
    pub context: Option<String>,
    pub user_id: Option<String>,
}

pub struct DashboardState {
    pub stats: DashboardStats,
    pub recent_analyses: Vec<AnalysisResult>,
    pub detector: crate::HallucinationDetector,
}

impl DashboardState {
    pub fn new() -> Self {
        Self {
            stats: DashboardStats {
                total_analyses: 0,
                average_hallucination_rate: 0.0,
                analyses_per_minute: 0.0,
                top_issues: Vec::new(),
                confidence_distribution: HashMap::new(),
            },
            recent_analyses: Vec::new(),
            detector: crate::HallucinationDetector::new(),
        }
    }
    
    pub fn update_stats(&mut self, result: &AnalysisResult) {
        self.stats.total_analyses += 1;
        self.recent_analyses.push(result.clone());
        
        // Keep only last 100 analyses
        if self.recent_analyses.len() > 100 {
            self.recent_analyses.remove(0);
        }
        
        // Update average hallucination rate
        let total_probability: f32 = self.recent_analyses.iter()
            .map(|r| r.hallucination_probability)
            .sum();
        self.stats.average_hallucination_rate = total_probability / self.recent_analyses.len() as f32;
        
        // Update confidence distribution
        let confidence_bucket = match result.confidence_score {
            x if x >= 0.9 => "Very High",
            x if x >= 0.7 => "High",
            x if x >= 0.5 => "Medium",
            x if x >= 0.3 => "Low",
            _ => "Very Low",
        };
        
        *self.stats.confidence_distribution.entry(confidence_bucket.to_string()).or_insert(0) += 1;
    }
}

pub async fn start_dashboard(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let state = Arc::new(Mutex::new(DashboardState::new()));
    
    // Clone for the stats updater
    let state_clone = Arc::clone(&state);
    
    // Background task to update stats
    tokio::spawn(async move {
        let mut interval = interval(Duration::from_secs(60));
        loop {
            interval.tick().await;
            // Update analyses per minute
            // Implementation depends on your needs
        }
    });
    
    // API Routes
    let analyze_route = warp::path("api")
        .and(warp::path("analyze"))
        .and(warp::post())
        .and(warp::body::json())
        .and(with_state(Arc::clone(&state)))
        .and_then(handle_analyze);
    
    let stats_route = warp::path("api")
        .and(warp::path("stats"))
        .and(warp::get())
        .and(with_state(Arc::clone(&state)))
        .and_then(handle_stats);
    
    let dashboard_route = warp::path::end()
        .and(warp::get())
        .and_then(serve_dashboard);
    
    let static_files = warp::path("static")
        .and(warp::fs::dir("static/"));
    
    let routes = analyze_route
        .or(stats_route)
        .or(dashboard_route)
        .or(static_files)
        .with(warp::cors().allow_any_origin());
    
    println!("🌐 Starting web dashboard on http://localhost:{}", port);
    warp::serve(routes)
        .run(([127, 0, 0, 1], port))
        .await;
    
    Ok(())
}

fn with_state(state: Arc<Mutex<DashboardState>>) -> impl Filter<Extract = (Arc<Mutex<DashboardState>>,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || Arc::clone(&state))
}

async fn handle_analyze(
    request: AnalysisRequest,
    state: Arc<Mutex<DashboardState>>,
) -> Result<impl warp::Reply, warp::Rejection> {
    let result = {
        let mut state_guard = state.lock().unwrap();
        let analysis = state_guard.detector.analyze_response(&request.text, request.context.as_deref());
        
        let result = AnalysisResult {
            hallucination_probability: analysis.hallucination_probability,
            confidence_score: analysis.confidence_score,
            detected_issues: analysis.detected_issues,
            recommendations: analysis.recommendations,
            user_id: request.user_id,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        
        state_guard.update_stats(&result);
        result
    };
    
    Ok(warp::reply::json(&result))
}

async fn handle_stats(
    state: Arc<Mutex<DashboardState>>,
) -> Result<impl warp::Reply, warp::Rejection> {
    let stats = {
        let state_guard = state.lock().unwrap();
        state_guard.stats.clone()
    };
    
    Ok(warp::reply::json(&stats))
}

async fn serve_dashboard() -> Result<impl warp::Reply, warp::Rejection> {
    Ok(warp::reply::html(include_str!("../templates/dashboard.html")))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub hallucination_probability: f32,
    pub confidence_score: f32,
    pub detected_issues: Vec<String>,
    pub recommendations: Vec<String>,
    pub user_id: Option<String>,
    pub timestamp: u64,
}
