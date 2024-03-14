# Option-Pricing

The work develops a dashboard that allows the user to perform option price estimation. It can be summarized as follows:
1. *User engagement*: The program prompts the user for inputs such as date, number of simulations, and option type.
2. *Stock modeling*: The selected stock undergoes fitting using either Geometric Brownian Motion or Variance Gamma stochastic processes.
3. *Parameter estimation*: Dedicated functions estimate parameters for both stochastic processes and calculate associated likelihood functions.
4. *Model selection*: The program uses the Akaike Criteria to choose the best-fitting stochastic process based on the observed data.
5. *Price simulation*: Based on the chosen process, the program generates a matrix of simulated prices to capture potential stock price movements over time.
6. *Option pricing*: Assuming estimation for a plain vanilla option, the program calculates the final price as the mean of the last column of the simulated price matrix, adjusted for risk neutrality compared to the strike price.

In the end, the dashboard shows the stock trend using a line graph, the performed simulations, and the estimated price. 
