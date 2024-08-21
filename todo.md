# todo

## Paradigm changes

- ### Heikin Ashi Candles - paradigm

  - use this candle paradigm for band_4
  - calculations:
    - very 1st candle, open (ha) = open

## Optimization changes

- ### combo callback

- ### robustness

  - shuffled test
    - add shuffling to win per days array and, take a compounding product, reduce it back to per day, then check 250 days value in simulation. and compare 10 such instances.

- ### more data

  - in win graph - check how far the avg pred_max was from true_max, when inside the band. same for min.

- ### 5 min data model

  - areas (1) - from start candle to 1200 candle, zone 2 - 1205 candle to last
  - 2 trades in a single day (2) - both 1m for 5m.
  - 10-11/11-12 and 12-13/13-14

     check feasibility

## Experiments

- ### feature engineering

  - features:
    - 5 min average
    - 15 min average
    - diff
    - lag features: -1m, -2m
    - 15 min standard deviation
    - vwap - volume weighted average price
    - vwap - volume weighted average price (5m)

## Data collection and stats

- get mean from open of 2nd zone. and from prev day close.
  - for min, max.
  - for both pred and real.

- intraday changes - stats:
  - check full day/ half day
  - graph
  - cumulative 30 days, always positive.

- ### correlation

  - correlation of all 50 stocks with each other.

  - correlation with nifty 50 index

- clear out the previous files folder

## Communication

- ### Mailgun

  - use mail gun to receive email of current status of you positions at intervals.

  - tutorial : [https://www.youtube.com/watch?v=LnVRGV-9NOY]

## Production

- ### API

  - use 3 brokers
