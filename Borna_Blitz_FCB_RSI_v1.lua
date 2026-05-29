-- ============================================================
--  STRATEGIJA: Borna Blitz FCB + RSI Binary v1
--  Platforma:  IQ Option (Lua)
--  Logika:     Fractal Chaos Bands wick sweep + RSI filter
--  Izlaz:      1 svecica (binary expiry)
-- ============================================================

instrument { name = "Borna Blitz FCB RSI", overlay = true }

-- ============================================================
-- 1. PODESAVANJA
-- ============================================================
fcb_period  = input(5,  "FCB Period",  input.integer, 2)   -- period za fractal prozor
rsi_period  = input(14, "RSI Period",  input.integer, 2)   -- period za RSI
rsi_ob      = input(60, "RSI Overbought", input.integer, 2) -- RSI granica za SELL filter
rsi_os      = input(40, "RSI Oversold",   input.integer, 2) -- RSI granica za BUY filter

-- ============================================================
-- 2. HARDKODIRANE BOJE (rgba() ne prikazuje hex tekst na grafiku)
-- ============================================================
col_upper = rgba(33,  150, 243, 1)   -- plava  - FCB gornja linija
col_lower = rgba(33,  150, 243, 1)   -- plava  - FCB donja linija
col_buy   = rgba(0,   200, 83,  1)   -- zelena - BUY signal
col_sell  = rgba(255, 23,  68,  1)   -- crvena - SELL signal
col_rsi   = rgba(255, 193, 7,   1)   -- zuta   - RSI linija (sub-panel)

-- ============================================================
-- 3. FRACTAL CHAOS BANDS (FCB)
--    Prozor = 2*period + 1 svecica
--    Fractal High: sredisnja svecica[period] je najvisa u prozoru
--    Fractal Low:  sredisnja svecica[period] je najniza u prozoru
--    value_when() drzi liniju ravnom do sledeceg fractala
-- ============================================================
window_size = 2 * fcb_period + 1

window_high = highest(high, window_size)
window_low  = lowest (low,  window_size)

is_fractal_high = high[fcb_period] >= window_high
is_fractal_low  = low[fcb_period]  <= window_low

fcb_upper = value_when(is_fractal_high, high[fcb_period], 1)
fcb_lower = value_when(is_fractal_low,  low[fcb_period],  1)

-- ============================================================
-- 4. RSI INDIKATOR
--    Klasicna Wilder-ova formula:
--    RS  = prosecni dobitak / prosecni gubitak (EMA glajding)
--    RSI = 100 - (100 / (1 + RS))
-- ============================================================
price_change = close - close[1]

gain = price_change > 0 and price_change or 0
loss = price_change < 0 and -price_change or 0

avg_gain = ema(gain, rsi_period)
avg_loss = ema(loss, rsi_period)

rs  = avg_loss ~= 0 and (avg_gain / avg_loss) or 100
rsi = 100 - (100 / (1 + rs))

-- ============================================================
-- 5. SIGNAL DETEKCIJA - Wick Sweep Reversal + RSI potvrda
--
--    BUY  (HIGHER):
--      - Prethodna svecica: Low[1] < fcb_lower[1]  (fitilj probio dno)
--      - Prethodna svecica: Close[1] > fcb_lower[1] (zatvorio se unutar)
--      - RSI[1] <= rsi_os  (podrucje preprodate imovine - oversold)
--
--    SELL (LOWER):
--      - Prethodna svecica: High[1] > fcb_upper[1]  (fitilj probio vrh)
--      - Prethodna svecica: Close[1] < fcb_upper[1] (zatvorio se unutar)
--      - RSI[1] >= rsi_ob  (podrucje prekupljene imovine - overbought)
-- ============================================================
wick_buy  = (low[1]  < fcb_lower[1]) and (close[1] > fcb_lower[1])
wick_sell = (high[1] > fcb_upper[1]) and (close[1] < fcb_upper[1])

buy_signal  = wick_buy  and (rsi[1] <= rsi_os)
sell_signal = wick_sell and (rsi[1] >= rsi_ob)

-- ============================================================
-- 6. CRTANJE FCB LINIJA NA GRAFIKONU (overlay)
-- ============================================================
plot(fcb_upper, "FCB Upper", col_upper, 2)
plot(fcb_lower, "FCB Lower", col_lower, 2)

-- ============================================================
-- 7. CRTANJE RSI U ZASEBNOM PANELU
-- ============================================================
indicator_separate()                              -- otvara sub-panel za RSI
plot(rsi,     "RSI",          col_rsi,  1)
plot(rsi_ob,  "Overbought",   col_sell, 1)        -- horizontalna linija OB
plot(rsi_os,  "Oversold",     col_buy,  1)        -- horizontalna linija OS

indicator_overlay()                               -- vraca crtanje na glavni grafik

-- ============================================================
-- 8. SIGNALNE STRELICE NA GLAVNOM GRAFIKONU
--    BUY  -> zelena strelica ISPOD svecice + tekst "HIGHER"
--    SELL -> crvena strelica IZNAD svecice + tekst "LOWER"
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
