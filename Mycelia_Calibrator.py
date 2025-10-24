def calibrate_from_logs(self, model_logs: list, target_model: str = 'mistral-7b') -> dict:
    """Empirical tune: Fit theorems to eval logs (e.g., contra_rate, self-query acc)."""
    if not model_logs:
        return {'error': 'No logs—run Mycelia first!'}
    
    # Thm1: Fit β from cap spikes (proxy: output len jumps)
    lens = [log.get('output_len', 50) for log in model_logs]
    jumps = np.diff(lens) > np.mean(np.diff(lens)) * 2  # Spike detect
    self.beta_exp = GLOBAL_CONSTANTS['NU_EXPONENT'] + 0.1 * np.sum(jumps) / len(lens)  # Bump on jumps
    
    # Thm2: C_AB^E from emergence rate (contra proxies)
    contra_rates = [log.get('contra_rate', 0.02) for log in model_logs]
    self.emergence_c = np.mean(contra_rates) * 2  # Scale to [0.8] baseline
    
    # Thm3: κ from self-acc decay (var as I_m proxy)
    i_ms = [log.get('i_m_proxy', 0.3) for log in model_logs]
    decays = [log.get('f_self', 0.9) for log in model_logs]
    kappa_fit = np.polyfit(i_ms, 1 - np.array(decays), 1)[0]  # Linear decay slope
    self.kappa = max(0.1, min(0.8, abs(kappa_fit)))  # Clamp sane
    
    # Thm4: Exponent from align drop (TruthfulQA proxy)
    aligns = [log.get('align_score', 0.8) for log in model_logs]  # Mock from evals
    params_proxy = len(model_logs) * 1e6  # Cycle as param stand-in
    exponent_fit = np.polyfit(np.log(params_proxy), np.log(np.maximum(aligns, 1e-6)), 1)[0]
    delta_align = GLOBAL_CONSTANTS['ALIGNMENT_DELTA_CAP'] + exponent_fit
    
    tuned = {
        'beta_exp': float(self.beta_exp),
        'emergence_c': float(self.emergence_c),
        'kappa': float(self.kappa),
        'delta_align': float(delta_align),
        'model': target_model,
        'calib_date': time.strftime('%Y-%m-%d')
    }
    print(f"Calibrated for {target_model}: β={tuned['beta_exp']:.3f}, C={tuned['emergence_c']:.3f}, κ={tuned['kappa']:.3f}, Δ_align={tuned['delta_align']:.3f}")
    return tuned  # Dump to JSONL for DeepSeek