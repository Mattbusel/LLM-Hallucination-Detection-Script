use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingChunk {
    pub id: String,
    pub text: String,
    pub timestamp: u64,
    pub is_complete: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingResult {
    pub chunk_id: String,
    pub cumulative_text: String,
    pub hallucination_probability: f32,
    pub confidence_trend: Vec<f32>,
    pub processing_time_ms: u64,
    pub word_count: usize,
}

pub struct StreamingHallucinationDetector {
    buffer: String,
    confidence_history: Vec<f32>,
    chunk_counter: usize,
    detector: crate::HallucinationDetector,
}

impl StreamingHallucinationDetector {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            confidence_history: Vec::new(),
            chunk_counter: 0,
            detector: crate::HallucinationDetector::new(),
        }
    }
    
    pub async fn process_stream(&mut self, 
        mut rx: mpsc::Receiver<StreamingChunk>
    ) -> mpsc::Receiver<StreamingResult> {
        let (tx, output_rx) = mpsc::channel(100);
        
        tokio::spawn(async move {
            while let Some(chunk) = rx.recv().await {
                let start_time = Instant::now();
                
                // Add chunk to buffer
                self.buffer.push_str(&chunk.text);
                self.chunk_counter += 1;
                
                // Analyze current buffer
                let result = self.detector.analyze_response(&self.buffer);
                let current_confidence = 1.0 - result.hallucination_probability;
                self.confidence_history.push(current_confidence);
                
                // Keep only last 10 confidence scores for trend
                if self.confidence_history.len() > 10 {
                    self.confidence_history.remove(0);
                }
                
                let processing_time = start_time.elapsed().as_millis() as u64;
                
                let streaming_result = StreamingResult {
                    chunk_id: chunk.id,
                    cumulative_text: self.buffer.clone(),
                    hallucination_probability: result.hallucination_probability,
                    confidence_trend: self.confidence_history.clone(),
                    processing_time_ms: processing_time,
                    word_count: self.buffer.split_whitespace().count(),
                };
                
                if tx.send(streaming_result).await.is_err() {
                    break;
                }
                
                if chunk.is_complete {
                    break;
                }
            }
        });
        
        output_rx
    }
    
    pub fn get_confidence_trend(&self) -> Vec<f32> {
        self.confidence_history.clone()
    }
    
    pub fn reset_buffer(&mut self) {
        self.buffer.clear();
        self.confidence_history.clear();
        self.chunk_counter = 0;
    }
}
