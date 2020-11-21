# Deep Hedging Demo

![Image of Demo](https://user-images.githubusercontent.com/7247589/99870023-ca5ec380-2b9d-11eb-8646-4e78ad87f8ad.png)

The Black-Scholes (BS) model – developed in 1973 and based on Nobel Prize winning works – has been the de-facto standard for pricing options and other financial derivatives for nearly half a century. The model can be used, under the assumption of a perfect financial market, to calculate an options price and the associated risk sensitivities. These risk sensitivities can then be theoretically used by a trader to create a perfect hedging strategy that eliminates all risks in a portfolio of options. However, the necessary conditions for a perfect financial market, such as zero transaction cost and the possibility of continuous trading, are difficult to meet in the real world. Therefore, in practice, banks have to rely on their traders’ intuition and experience to augment the BS model hedges with manual adjustments to account for these market imperfections.
The derivative desks of every bank all hedge their positions, and their PnL and risk exposure depend crucially on the quality of their hedges. If their hedges does not properly account for market imperfections, banks might underestimate the true risk exposure of their portfolios. On the other hand, if their hedges overestimate the cost of market imperfections, banks might overprice their positions (relative to their competitors) and hence risk losing trades and/or customers. Over the last few decades, the financial market has become increasingly sophisticated. Intuition and experience of traders might not be sufficiently fast and accurate to compute the impact of market imperfections on their portfolios and to come up with good manual adjustments to their BS model hedges. 

These limitations of the BS model are well-known, but neither academics nor practitioners have managed to develop alternatives to properly and systematically account for market frictions – at least not successful enough to be widely adopted by banks. Could machine learning (ML) be the cure? Last year, the Risk magazine reported that JP Morgan has begun to use machine learning to hedge (a.k.a. Deep Hedging) a portion of its vanilla index options flow book and plan to roll out the similar technology for single stocks, baskets and light exotics.  According to Risk.net (2019), the technology can create hedging strategies that “automatically factor in market fictions, such as transaction costs, liquidity constraints and risk limits”. More amazingly, the ML algorithm “far outperformed” hedging strategies derived from the BS model, and it could reduce the cost of hedging (in certain asset class) by “as much as 80%”. The technology has been heralded by some as “a breakthrough in quantitative finance, one that could mark the end of the Black-Scholes era.” Hence, it is not surprising that firms, such as Bank of America, Societe Generale and IBM, are reportedly developing their own ML-based system for derivative hedging.

Machine learning algorithms are often referred to as “black boxes” because of the inherent opaqueness and difficulties to inspect how an algorithm is able to accomplishing what is accomplishing. Buhler et al (2019) recently published a paper outlining the mechanism of this ground-breaking technology. We follow their outlined methodology to implement and replicate the “deep hedging” algorithm under different simulated market conditions. Given a distribution of the underlying assets and trader preference, the “deep hedging” algorithm attempts to identify the optimal hedge strategy (as a function of over 10k model parameters) that minimizes the residual risk of a hedged portfolio. We implement the “deep hedging” algorithm to demonstrate its potential benefit in a simplified yet sufficiently realistic setting. We first benchmark the deep hedging strategy against the classic Black-Scholes hedging strategy in a perfect world with no transaction cost, in which case the performance of both strategies should be similar. Then, we benchmark again in a world with market friction (i.e. non-zero transaction costs), in which case the deep hedging strategy should outperform the classic Black-Scholes hedging strategy. 

**References:**

Risk.net, (2019). “Deep hedging and the end of the Black-Scholes era.”

Hans Buhler et al, (2019). “Deep Hedging.” Quantitative Finance, 19(8).
