#include "market/market_events.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>

namespace deepquote {

// ============================================================================
// GBM Price Model Implementation
// ============================================================================

GBMPriceModel::GBMPriceModel(double mu, double sigma, double risk_free)
    : mu_(mu), sigma_(sigma), risk_free_rate_(risk_free) {
    
    std::random_device rd;
    rng_.seed(rd());
    normal_dist_ = std::normal_distribution<double>(0.0, 1.0);
}

double GBMPriceModel::generatePriceChange(double current_price, double dt) {
    // Geometric Brownian Motion: dS = S(μdt + σ√dt * Z)
    double drift = mu_ * dt;
    double diffusion = sigma_ * std::sqrt(dt) * normal_dist_(rng_);
    
    return current_price * (drift + diffusion);
}

void GBMPriceModel::updateParameters(const MarketEvent& event) {
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    
    switch (event.type) {
        case EventType::VOLATILITY_SPIKE:
            sigma_ *= (1.0 + event.magnitude * 2.0); // Increase volatility
            break;
        case EventType::MARKET_CRASH:
            mu_ -= event.magnitude * 0.1; // Negative drift
            sigma_ *= (1.0 + event.magnitude); // Higher volatility
            break;
        case EventType::NEWS_EVENT:
            // Random drift change based on news
            mu_ += (uniform_dist(rng) - 0.5) * event.magnitude * 0.2;
            break;
        case EventType::EARNINGS_ANNOUNCEMENT:
            // Significant drift change
            mu_ += (uniform_dist(rng) - 0.5) * event.magnitude * 0.5;
            sigma_ *= (1.0 + event.magnitude * 0.5); // Higher volatility
            break;
        default:
            break;
    }
}

void GBMPriceModel::reset() {
    // Reset to base parameters
    mu_ = 0.0;
    sigma_ = 0.2;
}

// ============================================================================
// Jump-Diffusion Model Implementation
// ============================================================================

JumpDiffusionModel::JumpDiffusionModel(double mu, double sigma, double lambda)
    : mu_(mu), sigma_(sigma), lambda_(lambda), jump_mu_(0.0), jump_sigma_(0.1) {
    
    std::random_device rd;
    rng_.seed(rd());
    normal_dist_ = std::normal_distribution<double>(0.0, 1.0);
    exp_dist_ = std::exponential_distribution<double>(lambda_);
}

double JumpDiffusionModel::generatePriceChange(double current_price, double dt) {
    // Standard GBM component
    double drift = mu_ * dt;
    double diffusion = sigma_ * std::sqrt(dt) * normal_dist_(rng_);
    
    // Jump component
    double jump = 0.0;
    double jump_probability = lambda_ * dt;
    
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    
    if (uniform_dist(rng) < jump_probability) {
        // Jump occurs
        jump = jump_mu_ + jump_sigma_ * normal_dist_(rng_);
    }
    
    return current_price * (drift + diffusion + jump);
}

void JumpDiffusionModel::updateParameters(const MarketEvent& event) {
    switch (event.type) {
        case EventType::PRICE_SHOCK:
            lambda_ *= (1.0 + event.magnitude * 3.0); // More jumps
            jump_sigma_ *= (1.0 + event.magnitude); // Larger jumps
            break;
        case EventType::FLASH_CRASH:
            lambda_ *= 10.0; // Many jumps
            jump_mu_ = -0.1; // Negative jumps
            jump_sigma_ *= 2.0; // Large jumps
            break;
        case EventType::VOLATILITY_SPIKE:
            sigma_ *= (1.0 + event.magnitude * 2.0);
            lambda_ *= (1.0 + event.magnitude);
            break;
        default:
            break;
    }
}

void JumpDiffusionModel::reset() {
    mu_ = 0.0;
    sigma_ = 0.2;
    lambda_ = 0.1;
    jump_mu_ = 0.0;
    jump_sigma_ = 0.1;
}

// ============================================================================
// Market Event Generator Implementation
// ============================================================================

MarketEventGenerator::MarketEventGenerator(const std::vector<std::string>& symbols)
    : symbols_(symbols), base_event_probability_(0.01), volatility_multiplier_(1.0), correlation_strength_(0.7) {
    
    std::random_device rd;
    rng_.seed(rd());
    uniform_dist_ = std::uniform_real_distribution<double>(0.0, 1.0);
    event_timing_dist_ = std::exponential_distribution<double>(base_event_probability_);
    
    // Initialize price models for each symbol
    for (const auto& symbol : symbols_) {
        // Use Jump-Diffusion for more realistic price movements
        price_models_[symbol] = std::make_unique<JumpDiffusionModel>(0.0, 0.2, 0.05);
    }
}

void MarketEventGenerator::update(double dt) {
    // Clear expired events
    clearExpiredEvents();
    
    // Check if we should generate a new event
    if (shouldGenerateEvent()) {
        EventType event_type = selectRandomEventType();
        std::string symbol = selectRandomSymbol();
        
        MarketEvent event = [&]() -> MarketEvent {
            switch (event_type) {
                case EventType::PRICE_SHOCK:
                    return generatePriceShock(symbol);
                case EventType::VOLATILITY_SPIKE:
                    return generateVolatilitySpike(symbol);
                case EventType::LIQUIDITY_CRISIS:
                    return generateLiquidityCrisis(symbol);
                case EventType::NEWS_EVENT:
                    return generateNewsEvent(symbol);
                case EventType::MARKET_CRASH:
                    return generateMarketCrash();
                case EventType::FLASH_CRASH:
                    return generateFlashCrash();
                case EventType::EARNINGS_ANNOUNCEMENT:
                    return generateEarningsAnnouncement(symbol);
                case EventType::FED_ANNOUNCEMENT:
                    return generateFedAnnouncement();
                case EventType::TECHNICAL_BREAKOUT:
                    return generateTechnicalBreakout(symbol);
                default:
                    return generateNewsEvent(symbol);
            }
        }();
        
        // Activate the event
        event.start_time = std::chrono::steady_clock::now();
        event.end_time = event.start_time + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(event.duration));
        event.is_active = true;
        
        // Add to active events
        active_events_.push_back(event);
        event_history_.push_back(event);
        
        // Update price models
        for (auto& [sym, model] : price_models_) {
            model->updateParameters(event);
        }
        
        std::cout << "Market Event: " << event.description << " (Magnitude: " 
                  << event.magnitude << ", Duration: " << event.duration << "s)" << std::endl;
    }
}

std::vector<MarketEvent> MarketEventGenerator::getActiveEvents() const {
    return active_events_;
}

void MarketEventGenerator::clearExpiredEvents() {
    auto now = std::chrono::steady_clock::now();
    active_events_.erase(
        std::remove_if(active_events_.begin(), active_events_.end(),
            [&](const MarketEvent& event) {
                return now > event.end_time;
            }),
        active_events_.end()
    );
}

double MarketEventGenerator::generatePriceChange(const std::string& symbol, double current_price, double dt) {
    auto it = price_models_.find(symbol);
    if (it != price_models_.end()) {
        return it->second->generatePriceChange(current_price, dt);
    }
    return 0.0;
}

// ============================================================================
// Event Generation Methods
// ============================================================================

MarketEvent MarketEventGenerator::generatePriceShock(const std::string& symbol) {
    double magnitude = generateEventMagnitude();
    double duration = generateEventDuration();
    std::string description = "Price shock for " + symbol + " - sudden large movement";
    
    return MarketEvent(EventType::PRICE_SHOCK, symbol, magnitude, duration, description);
}

MarketEvent MarketEventGenerator::generateVolatilitySpike(const std::string& symbol) {
    double magnitude = generateEventMagnitude();
    double duration = generateEventDuration() * 2.0; // Volatility spikes last longer
    std::string description = "Volatility spike for " + symbol + " - increased price swings";
    
    return MarketEvent(EventType::VOLATILITY_SPIKE, symbol, magnitude, duration, description);
}

MarketEvent MarketEventGenerator::generateLiquidityCrisis(const std::string& symbol) {
    double magnitude = generateEventMagnitude();
    double duration = generateEventDuration() * 3.0; // Liquidity crises last longer
    std::string description = "Liquidity crisis for " + symbol + " - reduced market depth";
    
    return MarketEvent(EventType::LIQUIDITY_CRISIS, symbol, magnitude, duration, description);
}

MarketEvent MarketEventGenerator::generateNewsEvent(const std::string& symbol) {
    double magnitude = generateEventMagnitude();
    double duration = generateEventDuration();
    
    std::vector<std::string> news_types = {
        "positive earnings guidance",
        "negative earnings guidance", 
        "merger announcement",
        "regulatory approval",
        "product launch",
        "management change",
        "analyst upgrade",
        "analyst downgrade"
    };
    
    std::string news_type = news_types[static_cast<int>(uniform_dist_(rng_) * news_types.size())];
    std::string description = "News event for " + symbol + ": " + news_type;
    
    return MarketEvent(EventType::NEWS_EVENT, symbol, magnitude, duration, description);
}

MarketEvent MarketEventGenerator::generateMarketCrash() {
    double magnitude = generateEventMagnitude() * 2.0; // Market crashes are stronger
    double duration = generateEventDuration() * 5.0; // Market crashes last longer
    std::string description = "Market crash - broad market decline affecting all symbols";
    
    return MarketEvent(EventType::MARKET_CRASH, "", magnitude, duration, description);
}

MarketEvent MarketEventGenerator::generateFlashCrash() {
    double magnitude = generateEventMagnitude() * 3.0; // Flash crashes are very strong
    double duration = 30.0; // Flash crashes are short but intense
    std::string description = "Flash crash - rapid market decline";
    
    return MarketEvent(EventType::FLASH_CRASH, "", magnitude, duration, description);
}

MarketEvent MarketEventGenerator::generateEarningsAnnouncement(const std::string& symbol) {
    double magnitude = generateEventMagnitude() * 1.5; // Earnings are significant
    double duration = generateEventDuration() * 2.0;
    std::string description = "Earnings announcement for " + symbol + " - quarterly results";
    
    return MarketEvent(EventType::EARNINGS_ANNOUNCEMENT, symbol, magnitude, duration, description);
}

MarketEvent MarketEventGenerator::generateFedAnnouncement() {
    double magnitude = generateEventMagnitude() * 1.5;
    double duration = generateEventDuration() * 3.0;
    std::string description = "Federal Reserve announcement - monetary policy changes";
    
    return MarketEvent(EventType::FED_ANNOUNCEMENT, "", magnitude, duration, description);
}

MarketEvent MarketEventGenerator::generateTechnicalBreakout(const std::string& symbol) {
    double magnitude = generateEventMagnitude();
    double duration = generateEventDuration() * 1.5;
    std::string description = "Technical breakout for " + symbol + " - price breaks key levels";
    
    return MarketEvent(EventType::TECHNICAL_BREAKOUT, symbol, magnitude, duration, description);
}

// ============================================================================
// Helper Methods
// ============================================================================

bool MarketEventGenerator::shouldGenerateEvent() const {
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    return uniform_dist(rng) < base_event_probability_;
}

EventType MarketEventGenerator::selectRandomEventType() const {
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    
    std::vector<EventType> event_types = {
        EventType::PRICE_SHOCK,
        EventType::VOLATILITY_SPIKE,
        EventType::NEWS_EVENT,
        EventType::EARNINGS_ANNOUNCEMENT,
        EventType::TECHNICAL_BREAKOUT,
        EventType::LIQUIDITY_CRISIS,
        EventType::MARKET_CRASH,
        EventType::FLASH_CRASH,
        EventType::FED_ANNOUNCEMENT
    };
    
    return event_types[static_cast<int>(uniform_dist(rng) * event_types.size())];
}

std::string MarketEventGenerator::selectRandomSymbol() const {
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    return symbols_[static_cast<int>(uniform_dist(rng) * symbols_.size())];
}

double MarketEventGenerator::generateEventMagnitude() const {
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    
    // Use a power law distribution for event magnitudes (small events more common)
    double u = uniform_dist(rng);
    return std::pow(u, 2.0); // Bias toward smaller events
}

double MarketEventGenerator::generateEventDuration() const {
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    
    // Events last between 30 seconds and 5 minutes
    return 30.0 + uniform_dist(rng) * 270.0;
}

std::string MarketEventGenerator::generateEventDescription(EventType type, const std::string& symbol) const {
    // This is handled in the specific event generation methods
    return "";
}

// ============================================================================
// Microstructure Noise Implementation
// ============================================================================

MicrostructureNoise::MicrostructureNoise(double amplitude, double mean_reversion)
    : noise_amplitude_(amplitude), mean_reversion_speed_(mean_reversion) {
    
    std::random_device rd;
    rng_.seed(rd());
    noise_dist_ = std::normal_distribution<double>(0.0, 1.0);
}

double MicrostructureNoise::generateNoise(double dt) {
    // Generate mean-reverting noise
    static double current_noise = 0.0;
    
    // Mean reversion component
    current_noise -= mean_reversion_speed_ * current_noise * dt;
    
    // Random component
    current_noise += noise_amplitude_ * std::sqrt(dt) * noise_dist_(rng_);
    
    return current_noise;
}

} // namespace deepquote 