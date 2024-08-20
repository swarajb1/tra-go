# notes

## tricks

- in simulation.
  - if expected_reward_percent < 0.05 %:
        no trade that day.

    - ```python
        # special_condition
        if expected_reward / ((buy_price + sell_price) / 2) * 100 < 0.05:
            # reward is less than 0.05% of the invested amount
            # so, not worth trading
            continue
      ```

## other info

- find out 5 day hl from some amount of input data. (5 days or may be 3 months)
  - capture changes that occur outside of given time-frame
  - the close and open

- learnings gpt:
  - **Scenario 3: Temperature Prediction:**
    - **Problem Description:** You are building a model to predict daily temperatures based on features like humidity, wind speed, and historical temperature data.
    - **Loss Function Choice:**
      - **MSE:** If you choose MSE, the model will be more sensitive to days where the temperature predictions deviate significantly from the actual temperatures. This might be suitable if you want to prioritize accurate predictions for days with extreme temperatures.
      - **MAE:** If you use MAE, the model will treat all temperature prediction errors equally, making it less sensitive to outliers. This could be appropriate if you're more concerned about overall accuracy across a range of temperatures.
  -
  - **Scenario 4: Stock Price Prediction:**
    - **Problem Description:** Building a model to predict daily stock prices based on historical stock data, trading volume, and other relevant features.
    - **Loss Function Choice:**
      - **MSE:** MSE might be chosen if you want the model to be more sensitive to significant deviations in stock price predictions, especially during periods of high volatility. It penalizes larger errors more heavily.
      - **MAE:** MAE could be chosen if you want a more robust model that is less influenced by extreme stock price fluctuations.

- 1. **Feature Engineering**: This involves creating new features from the existing data. For example, you could create features that capture the change in price from one day to the next, or the average price over the last few days.
