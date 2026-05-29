-- ============================================================
--  STRATEGIJA: Borna Blitz FCB + RSI Binary v3
--  Platforma:  IQ Option / QuadCode Script (Lua)
--  FIX v3:
--    1. plot_shape ima 7 argumenata MAX - uklonjen 8. (text color)
--    2. FCB logika: value_when radi sa serijama, ne get_value()
--    3. Signal detekcija: get_value() samo za finalne skalarne provjere
-- ============================================================

instrument { name = "Borna Blitz FCB RSI v3", overlay = true }

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
--    value_when() mora primiti SERIJU kao uslov, ne scalar.
--    Koristimo serijske poređenje: highest/lowest su serije.
--    high == highest(high, N) je true samo na baru koji je novi max.
-- ============================================================
window_size = 2 * fcb_period + 1

-- Serija: true na svakom baru koji je lokalni fractal vrh/dno
is_fractal_high = high == highest(high, window_size)
is_fractal_low  = low  == lowest (low,  window_size)

-- value_when drzi vrednost ravnom dok se ne pojavi novi fractal
fcb_upper = value_when(is_fractal_high, high, 1)
fcb_lower = value_when(is_fractal_low,  low,  1)

-- ============================================================
-- 4. RSI - ugradjena funkcija
-- ============================================================
rsi_line = rsi(close, rsi_period)

-- ============================================================
-- 5. SIGNAL DETEKCIJA
--    get_value() za skalarne vrednosti prethodne svecice
-- ============================================================
prev_low    = get_value(low,       1)
prev_high   = get_value(high,      1)
prev_close  = get_value(close,     1)
prev_upper  = get_value(fcb_upper, 1)
prev_lower  = get_value(fcb_lower, 1)
prev_rsi    = get_value(rsi_line,  1)

wick_buy  = (prev_low  < prev_lower) and (prev_close > prev_lower)
wick_sell = (prev_high > prev_upper) and (prev_close < prev_upper)

buy_signal  = wick_buy  and (prev_rsi <= rsi_os)
sell_signal = wick_sell and (prev_rsi >= rsi_ob)

-- ============================================================
-- 6. CRTANJE FCB LINIJA
-- ============================================================
plot(fcb_upper, "FCB Upper", col_upper, 2)
plot(fcb_lower, "FCB Lower", col_lower, 2)

-- ============================================================
-- 7. RSI U ZASEBNOM PANELU
-- ============================================================
indicator_separate()
plot(rsi_line, "RSI",        col_rsi,  1)
plot(rsi_ob,   "Overbought", col_sell, 1)
plot(rsi_os,   "Oversold",   col_buy,  1)
indicator_overlay()

-- ============================================================
-- 8. SIGNALNE STRELICE
--    FIX: 7 argumenata max - uklonjen 8. argument (text color)
--         koji je uzrokovao prikaz hex teksta na grafiku
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
