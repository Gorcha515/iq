-- ====================================================
--  STRATEGIJA: Borna Blitz Chaos Reversal v4
--  FIX: Hardkodirane rgba() boje  bez input.color
--       koji uzrokuje prikaz hex teksta
-- ====================================================

instrument { name = "Borna Blitz Chaos Reversal", overlay = true }

--
-- 1. PODEAVANJA  samo period, bez color inputa
--
period = input(5, "FCB Period", input.integer, 2)

--
-- 2. HARDKODIRANE BOJE (rgba ne uzrokuje bug)
--
col_upper = rgba(33,  150, 243, 1)   -- plava   FCB gornja linija
col_lower = rgba(33,  150, 243, 1)   -- plava   FCB donja linija
col_buy   = rgba(0,   200, 83,  1)   -- zelena  BUY strelica
col_sell  = rgba(255, 23,  68,  1)   -- crvena  SELL strelica

--
-- 3. FRACTAL CHAOS BANDS
--    Fractal High: sveica[period] = max u prozoru
--    Fractal Low:  sveica[period] = min u prozoru
--    value_when() dri liniju do sljedeeg fractala
--
window_high = highest(high, 2 * period + 1)
window_low  = lowest (low,  2 * period + 1)

is_fractal_high = high[period] >= window_high
is_fractal_low  = low[period]  <= window_low

fcb_upper = value_when(is_fractal_high, high[period], 1)
fcb_lower = value_when(is_fractal_low,  low[period],  1)

--
-- 4. SIGNAL DETEKCIJA  Wick Sweep Reversal
--
--    BUY:  Low[1]   < fcb_lower[1]
--          Close[1] > fcb_lower[1]
--           fitilj probio dno, zatvorio se unutar
--
--    SELL: High[1]  > fcb_upper[1]
--          Close[1] < fcb_upper[1]
--           fitilj probio vrh, zatvorio se unutar
--
buy_signal  = (low[1]  < fcb_lower[1]) and (close[1] > fcb_lower[1])
sell_signal = (high[1] > fcb_upper[1]) and (close[1] < fcb_upper[1])

--
-- 5. CRTANJE FCB LINIJA NA GRAFIKONU
--
plot(fcb_upper, "FCB Upper", col_upper, 2)
plot(fcb_lower, "FCB Lower", col_lower, 2)

--
-- 6. SIGNALNE STRELICE
--    shape_style.arrowup    zelena strelica gore
--    shape_style.arrowdown  crvena strelica dolje
--
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
