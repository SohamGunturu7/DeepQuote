#pragma once

#include <random>
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <chrono>

namespace deepquote {

// ============================================================================
// Market Event Types
// ============================================================================

enum class EventType {
    PRICE_SHOCK,           // Sudden large price movement
    VOLATILITY_SPIKE,      // Increased price volatility
    LIQUIDITY_CRISIS,      // Reduced market liquidity
    NEWS_EVENT,           // Company-specific news
    MARKET_CRASH,         // Broad market decline
    FLASH_CRASH,          // Very rapid price decline
    PUMP_AND_DUMP,        // Artificial price manipulation
    EARNINGS_ANNOUNCEMENT, // Quarterly earnings
    FED_ANNOUNCEMENT,     // Central bank policy changes
    TECHNICAL_BREAKOUT,   // Price breaks key levels
    CORRELATION_BREAKDOWN, // Assets become uncorrelated
    MICROSTRUCTURE_NOISE  // Small random movements
};

// ============================================================================
// Market Event Structure
// ============================================================================

struct MarketEvent {
    EventType type;
    std::string symbol;
    double magnitude;           // How strong the event is (0-1)
    double duration;            // How long the event lasts (seconds)
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    std::string description;
    bool is_active;
    
    MarketEvent(EventType t, const std::string& sym, double mag, double dur, const std::string& desc)
        : type(t), symbol(sym), magnitude(mag), duration(dur), description(desc), is_active(false) {}
};

// ============================================================================
// Price Movement Models
// ============================================================================

class PriceMovementModel {
public:
    virtual ~PriceMovementModel() = default;
    virtual double generatePriceChange(double current_price, double dt) = 0;
    virtual void updateParameters(const MarketEvent& event) = 0;
    virtual void reset() = 0;
};

// Geometric Brownian Motion (Standard model)
class GBMPriceModel : public PriceMovementModel {
private:
    double mu_;           // Drift (expected return)
    double sigma_;        // Volatility
    double risk_free_rate_;
    std::mt19937 rng_;
    std::normal_distribution<double> normal_dist_;
    
public:
    GBMPriceModel(double mu = 0.0, double sigma = 0.2, double risk_free = 0.02);
    double generatePriceChange(double current_price, double dt) override;
    void updateParameters(const MarketEvent& event) override;
    void reset() override;
    
    void setVolatility(double sigma) { sigma_ = sigma; }
    void setDrift(double mu) { mu_ = mu; }
    double getVolatility() const { return sigma_; }
    double getDrift() const { return mu_; }
};

// Jump-Diffusion Model (Handles sudden jumps)
class JumpDiffusionModel : public PriceMovementModel {
private:
    double mu_;           // Drift
    double sigma_;        // Volatility
    double lambda_;       // Jump intensity
    double jump_mu_;      // Jump size mean
    double jump_sigma_;   // Jump size volatility
    std::mt19937 rng_;
    std::normal_distribution<double> normal_dist_;
    std::exponential_distribution<double> exp_dist_;
    
public:
    JumpDiffusionModel(double mu = 0.0, double sigma = 0.2, double lambda = 0.1);
    double generatePriceChange(double current_price, double dt) override;
    void updateParameters(const MarketEvent& event) override;
    void reset() override;
};

// ============================================================================
// Market Event Generator
// ============================================================================

class MarketEventGenerator {
private:
    std::vector<std::string> symbols_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_dist_;
    std::exponential_distribution<double> event_timing_dist_;
    
    // Event probabilities and parameters
    double base_event_probability_;
    double volatility_multiplier_;
    double correlation_strength_;
    
    // Event history
    std::vector<MarketEvent> active_events_;
    std::vector<MarketEvent> event_history_;
    
    // Price models for each symbol
    std::unordered_map<std::string, std::unique_ptr<PriceMovementModel>> price_models_;
    
public:
    MarketEventGenerator(const std::vector<std::string>& symbols);
    
    // Event generation
    void update(double dt);
    std::vector<MarketEvent> getActiveEvents() const;
    void clearExpiredEvents();
    
    // Price movement generation
    double generatePriceChange(const std::string& symbol, double current_price, double dt);
    
    // Event-specific methods
    MarketEvent generatePriceShock(const std::string& symbol);
    MarketEvent generateVolatilitySpike(const std::string& symbol);
    MarketEvent generateLiquidityCrisis(const std::string& symbol);
    MarketEvent generateNewsEvent(const std::string& symbol);
    MarketEvent generateMarketCrash();
    MarketEvent generateFlashCrash();
    MarketEvent generateEarningsAnnouncement(const std::string& symbol);
    MarketEvent generateFedAnnouncement();
    MarketEvent generateTechnicalBreakout(const std::string& symbol);
    
    // Configuration
    void setBaseEventProbability(double prob) { base_event_probability_ = prob; }
    void setVolatilityMultiplier(double mult) { volatility_multiplier_ = mult; }
    void setCorrelationStrength(double strength) { correlation_strength_ = strength; }
    
    // Statistics
    size_t getActiveEventCount() const { return active_events_.size(); }
    size_t getTotalEventCount() const { return event_history_.size(); }
    
private:
    bool shouldGenerateEvent() const;
    EventType selectRandomEventType() const;
    std::string selectRandomSymbol() const;
    double generateEventMagnitude() const;
    double generateEventDuration() const;
    std::string generateEventDescription(EventType type, const std::string& symbol) const;
};

// ============================================================================
// Market Microstructure Noise
// ============================================================================

class MicrostructureNoise {
private:
    double noise_amplitude_;
    double mean_reversion_speed_;
    std::mt19937 rng_;
    std::normal_distribution<double> noise_dist_;
    
public:
    MicrostructureNoise(double amplitude = 0.001, double mean_reversion = 0.1);
    double generateNoise(double dt);
    void setAmplitude(double amplitude) { noise_amplitude_ = amplitude; }
};

} // namespace deepquote 