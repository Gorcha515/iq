-- ============================================================
--  STRATEGIJA: Borna Blitz FCB + RSI Binary v2
--  Platforma:  IQ Option / QuadCode Script (Lua)
--  FIX v2:
--    1. Sve [1] indekse zamenjeno sa get_value(serija, 1)
--    2. Rucni RSI racun zamenjen ugradenom rsi() funkcijom
--    3. FCB fractal indeksi takodje kroz get_value()
-- ============================================================

instrument { name = "Borna Blitz FCB RSI v2", overlay = true }

-- ============================================================
-- 1. PODESAVANJA
-- ============================================================
fcb_period = input(5,  "FCB Period",      input.integer, 2)
rsi_period = input(14, "RSI Period",      input.integer, 2)
rsi_ob     = input(60, "RSI Overbought",  input.integer, 2)
rsi_os     = input(40, "RSI Oversold",    input.integer, 2)

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
--    FIX: get_value() za pouzdano citanje istorijskih barova
-- ============================================================
window_size = 2 * fcb_period + 1

window_high = highest(high, window_size)
window_low  = lowest(low,   window_size)

-- get_value(serija, offset) - ispravan nacin za istoriju u QuadCode
is_fractal_high = get_value(high, fcb_period) >= get_value(window_high, fcb_period)
is_fractal_low  = get_value(low,  fcb_period) <= get_value(window_low,  fcb_period)

fcb_upper = value_when(is_fractal_high, get_value(high, fcb_period), 1)
fcb_lower = value_when(is_fractal_low,  get_value(low,  fcb_period), 1)

-- ============================================================
-- 4. RSI - ugradjena funkcija umesto rucne formule
--    FIX: rsi(source, period) je stabilan i ne uzrokuje
--         gresku sa and/or nad serijama podataka
-- ============================================================
rsi_line = rsi(close, rsi_period)

-- ============================================================
-- 5. SIGNAL DETEKCIJA - Wick Sweep Reversal + RSI potvrda
--    FIX: get_value() umesto [1] za sve istorijske serije
--
--    BUY  (HIGHER):
--      - get_value(low, 1)   < get_value(fcb_lower, 1)  fitilj probio dno
--      - get_value(close, 1) > get_value(fcb_lower, 1)  zatvorio se unutar
--      - get_value(rsi_line, 1) <= rsi_os               oversold zona
--
--    SELL (LOWER):
--      - get_value(high, 1)  > get_value(fcb_upper, 1)  fitilj probio vrh
--      - get_value(close, 1) < get_value(fcb_upper, 1)  zatvorio se unutar
--      - get_value(rsi_line, 1) >= rsi_ob               overbought zona
-- ============================================================
prev_low   = get_value(low,       1)
prev_high  = get_value(high,      1)
prev_close = get_value(close,     1)
prev_upper = get_value(fcb_upper, 1)
prev_lower = get_value(fcb_lower, 1)
prev_rsi   = get_value(rsi_line,  1)

wick_buy  = (prev_low   < prev_lower) and (prev_close > prev_lower)
wick_sell = (prev_high  > prev_upper) and (prev_close < prev_upper)

buy_signal  = wick_buy  and (prev_rsi <= rsi_os)
sell_signal = wick_sell and (prev_rsi >= rsi_ob)

-- ============================================================
-- 6. CRTANJE FCB LINIJA NA GRAFIKONU
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
-- ============================================================
plot_shape(
    buy_signal,
    shape_style.arrowup,
    shape_location.belowbar,
    shape_size.large,
    col_buy,
    "BUY",
    "HIGHER",
    col_buy
)

plot_shape(
    sell_signal,
    shape_style.arrowdown,
    shape_location.abovebar,
    shape_size.large,
    col_sell,
    "SELL",
    "LOWER",
    col_sell
)
