-- ============================================================
--  STRATEGIJA: Borna Blitz FCB + RSI Binary v4
--  Platforma:  IQ Option / QuadCode Script (Lua)
--  FIX v4:
--    Dodata potvrda sledecim barom:
--      SELL: bar[2] pravio sweep vrha + bar[1] zatvorio crveno
--      BUY:  bar[2] pravio sweep dna  + bar[1] zatvorio zeleno
--    Ovo eliminiše signale u trendu gde svaka nova svecica
--    izgleda kao reversal bez prave potvrde.
-- ============================================================

instrument { name = "Borna Blitz FCB RSI v4", overlay = true }

-- ============================================================
-- 1. PODESAVANJA
-- ============================================================
fcb_period = input(5,  "FCB Period",     input.integer, 2)
rsi_period = input(14, "RSI Period",     input.integer, 2)
rsi_ob     = input(60, "RSI Overbought", input.integer, 2)
rsi_os     = input(40, "RSI Oversold",   input.integer, 2)

-- ============================================================
-- 2. HARDKODIRANE BOJE
-- ============================================================
col_upper = rgba(33,  150, 243, 1)
col_lower = rgba(33,  150, 243, 1)
col_buy   = rgba(0,   200, 83,  1)
col_sell  = rgba(255, 23,  68,  1)
col_rsi   = rgba(255, 193, 7,   1)

-- ============================================================
-- 3. FRACTAL CHAOS BANDS
--    is_fractal_high/low su serije (true/false po baru)
--    value_when zadrzava vrednost do sledeceg fractala
-- ============================================================
window_size = 2 * fcb_period + 1

is_fractal_high = high == highest(high, window_size)
is_fractal_low  = low  == lowest (low,  window_size)

fcb_upper = value_when(is_fractal_high, high, 1)
fcb_lower = value_when(is_fractal_low,  low,  1)

-- ============================================================
-- 4. RSI
-- ============================================================
rsi_line = rsi(close, rsi_period)

-- ============================================================
-- 5. ISTORIJSKE VREDNOSTI
--    bar[1] = prethodna svecica (potvrdna)
--    bar[2] = svecica pre nje   (sweep svecica)
-- ============================================================

-- Potvrdna svecica (bar[1])
conf_close  = get_value(close, 1)
conf_open   = get_value(open,  1)

-- Sweep svecica (bar[2])
sweep_low   = get_value(low,   2)
sweep_high  = get_value(high,  2)
sweep_close = get_value(close, 2)

-- FCB vrednosti na sweep svecici (bar[2])
upper_at_sweep = get_value(fcb_upper, 2)
lower_at_sweep = get_value(fcb_lower, 2)

-- RSI na sweep svecici (bar[2])
rsi_at_sweep = get_value(rsi_line, 2)

-- ============================================================
-- 6. SIGNAL DETEKCIJA SA POTVRDOM
--
--    BUY  (HIGHER):
--      bar[2]: Low probio FCB dno, Close zatvorio unutar  <- sweep
--      bar[1]: Close > Open  (zelena potvrdna svecica)    <- potvrda
--      RSI na bar[2] <= rsi_os                            <- oversold
--
--    SELL (LOWER):
--      bar[2]: High probio FCB vrh, Close zatvorio unutar <- sweep
--      bar[1]: Close < Open  (crvena potvrdna svecica)    <- potvrda
--      RSI na bar[2] >= rsi_ob                            <- overbought
-- ============================================================
sweep_buy  = (sweep_low   < lower_at_sweep) and (sweep_close > lower_at_sweep)
sweep_sell = (sweep_high  > upper_at_sweep) and (sweep_close < upper_at_sweep)

confirm_bull = conf_close > conf_open   -- zelena svecica
confirm_bear = conf_close < conf_open   -- crvena svecica

buy_signal  = sweep_buy  and confirm_bull and (rsi_at_sweep <= rsi_os)
sell_signal = sweep_sell and confirm_bear and (rsi_at_sweep >= rsi_ob)

-- ============================================================
-- 7. CRTANJE FCB LINIJA
-- ============================================================
plot(fcb_upper, "FCB Upper", col_upper, 2)
plot(fcb_lower, "FCB Lower", col_lower, 2)

-- ============================================================
-- 8. RSI U ZASEBNOM PANELU
-- ============================================================
indicator_separate()
plot(rsi_line, "RSI",        col_rsi,  1)
plot(rsi_ob,   "Overbought", col_sell, 1)
plot(rsi_os,   "Oversold",   col_buy,  1)
indicator_overlay()

-- ============================================================
-- 9. SIGNALNE STRELICE (7 argumenata - bez text color)
-- ============================================================
plot_shape(
    buy_signal,
    shape_style.arrowup,
    shape_location.belowbar,
    shape_size.large,
    col_buy,
    "BUY",
    "HIGHER"
)

plot_shape(
    sell_signal,
    shape_style.arrowdown,
    shape_location.abovebar,
    shape_size.large,
    col_sell,
    "SELL",
    "LOWER"
)
