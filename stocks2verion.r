library(quantmod)
library(ggplot2)
library(forecast)
library(tseries)
library(rugarch)
library(dplyr)
library(Hmisc)
library(moments)
library(urca)
library(TSA)
library(lmtest)
library(aTSA)
library(fGarch)
library(SBAGM)

# getting data from yahoo.finance!
print(getSymbols("^GSPC", src = "yahoo", from = "2000-01-01", to = "2020-10-01"))

# ajdusting data for non-stationarity usinf differencing and log
gspc <- diff(log(GSPC$GSPC.Adjusted))
gspc <- na.omit(gspc)

# saving data for lstm
# write.table(gspc,
#     file = "data.csv",
#     sep = "\t", row.names = T
# )

# training sample is 5133 observations, the leftover is testing sample
gspc_train <- gspc[1:5133]
print(head(gspc_train))
gspc_test <- gspc[5133:length(gspc)]


# plotting hist and normalized line
# hist(gspc_train, freq = F)
# x <- seq(-15, +15, by = 0.02) #range of values are min&max for x-axis of normal line
# curve(dnorm(x), add = TRUE)

# descriptive stats
# use describe(), var(), median(), sd(), scewness(), kurtosis() for describptive statistics
print(jarque.bera.test(gspc_train))

# stationary tests
print(adf.test(gspc_train))
print(kpss.test(gspc_train))
print(summary(ur.ers(gspc_train, type = "P-test")))

# autocorrelation tests and graphs
print(Box.test(gspc_train, lag = 24, type = "Ljung"))

# Computes the sample extended acf (ESACF) for the time series stored in z. The matrix of
# ESACF with the AR order up to ar.max and the MA order up to ma.max is stored in the
# matrix EACFM.
print(eacf(gspc_train, ar.max = 5, ma.max = 5))

# chart acf(z), acf(abs(z)), acf(z^2), pacf(z) autocorrelation and
# partial autocorrelation graphs
mx <- 40
acf((gspc_train)^2, lag.max = mx)
axis(1, at = 0:mx / 12, labels = 0:mx)

# model building, ARIMA
fit_arima_model <- auto.arima(gspc_train,
    stationary = TRUE, ic = "aic",
    trace = TRUE
)
print(summary(fit_arima_model))

print(coeftest(arima(gspc_train, order = c(1, 0, 0))))
print(accuracy(arima(gspc_train, order = c(1, 0, 0))))
print(t.test(gspc_train)) # Ho: mu != 0

# arch effect test, portomanteau Q and lagrange multiplier tests, graphs
# null hypothesis: residuals of ARIMA mmodel are homoscedastic
residuals <- gspc_train - mean(gspc_train)
checkresiduals(fit_arima_model)

Resids <- fit_arima_model$residuals
# plot(Resids)

par(mfrow = c(1, 2))
acf(residuals^2)
pacf(residuals^2)


# choosing best arima using loop and p+q=10, p,q <=5
# storing it as a dataframe(), alternative way to determine the prosess's order.
Result <- data.frame(Model = "m", AIC = 0)
q <- 0
for (i in 1:3) {
    for (j in 1:3) {
        q <- q + 1
        fit <- garchFit(substitute(~ garch(p, q), list(p = i, q = j)), data = Resids, trace = F)
        # print(fit)
        Result <- rbind(Result, data.frame(Model = paste("m-", i, "-", j), AIC = fit@fit$ics[1]))
    }
}

# ugarch arguments and using loop to test various p+q<=10 combinations
# for GARCH models, change model to "model=rGARCH" (for standard garch), "eGARCH"(exponential),
# and "fGARCH" (omnibus model); also tried "apARCH" (also omnibus model, some overlap with "fGARCH")

Result2 <- data.frame(Model = "e", AIC = 0)

for (p in 1:3) {
    for (q in 1:3) {
        spec <- ugarchspec(
            variance.model = list(
                model = "eGARCH",
                garchOrder = c(p, q),
                submodel = NULL,
                external.regressors = NULL,
                variance.targeting = FALSE
            ),
            mean.model = list(
                armaOrder = c(1, 0),
                include.mean = TRUE
            ),
            distribution.model = "sstd"
        )
        fit <- ugarchfit(spec, gspc_train,
            solver = "hybrid"
        )
        Result2 <- rbind(Result2, data.frame(Model = paste("ap-", p, "-", q), AIC = infocriteria(fit)[1]))

        # print(infocriteria(fit)[1])
    }
}
print(Result2)

# again, testing various model specifications: eGARCH, order c=(2,1):
#                                               apARCH, order c=(1,1)
spec <- ugarchspec(
    variance.model = list(
        model = "eGARCH",
        garchOrder = c(2, 1),
        submodel = NULL,
        external.regressors = NULL,
        variance.targeting = FALSE
    ),
    mean.model = list(
        armaOrder = c(1, 0),
        include.mean = TRUE
    ),
    distribution.model = "sstd"
)


fit <- ugarchfit(spec, gspc_train,
    solver = "hybrid"
)
print(fit)
print(persistence(fit)) # Persistence of valatility
print(convergence(fit))
print(likelihood(fit))
print(coef(fit))
print(nyblom(fit))

# graphical representations of varioius caracteristics/parametrizations of chosen models
png(filename = "C:/Users/nada/OneDrive/Documents/Desktop/model1.png")
par(mfrow = c(2, 2))
plot(fit, which = 1)
plot(fit, which = 2)
plot(fit, which = 3)
plot(fit, which = 4)
dev.off()

png(filename = "C:/Users/nada/OneDrive/Documents/Desktop/model2.png")
par(mfrow = c(2, 2))
plot(fit, which = 5)
plot(fit, which = 6)
plot(fit, which = 7)
plot(fit, which = 8)
dev.off()

png(filename = "C:/Users/nada/OneDrive/Documents/Desktop/model3.png")
par(mfrow = c(2, 2))
plot(fit, which = 9)
plot(fit, which = 10)
plot(fit, which = 11)
plot(fit, which = 12)
dev.off()

# :linear: 20 steps forecast; not good, next try rolling forecast
forcast1 <- ugarchforecast(fit, data = gspc_train, n.ahead = 20)
print(forcast1)

# comparing two best models: egarch and aparch
forecast_rsma <- appgarch(gspc,
    methods = c("eGARCH", "apARCH"),
    distributions = "sstd", aorder = c(1, 0),
    gorder = c(2, 1), algo = "hybrid", stepahead = 5
)
print(forecast_rsma)

# our final model: specifications, fitting the data and model
# giving at the end moving, rolling forecast, out of sample
spec100 <- ugarchspec(
    variance.model = list(model = "eGARCH", garchOrder = c(2, 1)),
    mean.model = list(armaOrder = c(1, 0), include.mean = TRUE),
    distribution.model = "sstd"
)
model100 <- ugarchfit(spec100, gspc_train, solver = "hybrid")
print(model100)
print("a")
# g(1,1)a(0,0)~64%


# forcast1 <- rep(0, times = length(gspc_test))
# for (i in 1:length(gspc_test)) {
#     ystar_exp <- gspc[1:(length(gspc_test - 1) + i)]
#     forecast_model1 <- ugarchfit(spec100, ystar_exp, solver = "hybrid")
#     for_obj <- ugarchforecast(forecast_model1, n.ahead = 1)
#     # print(for_obj)
#     forcast1[i] <- for_obj@forecast$sigmaFor[1]
# }
# print(head(forcast1))
print("b")

# # png(filename = "C:/Users/nada/OneDrive/Documents/Desktop/model3.png")
# # par(mfrow = c(1, 2))
# # print(forcast1)
# # print(forcast1[1])
# # print(forcast1[2])
# # dev.off()

# fit_roll <- ugarchfit(spec100, gspc, out.sample = 85)
# fore_roll <- ugarchforecast(fit_roll, n.ahead = 20, n.roll = 50)
# # print(fore_roll)

# # png(filename = "C:/Users/nada/OneDrive/Documents/Desktop/model4.png")
# # # par(mfrow = c(1, 2))
# # plot(sigma(fore_roll))
# # dev.off()
# # print(accuracy(fore_roll))


# we are not refitting the model; in order to use "moving" rolling forecast, we set
# refit step n=length(test data)-1, o refit happens at the last day of our testing sample
# so the function can do it's "magic", but the model has not been refitted to the newly
# available data.
garchroll <- ugarchroll(spec100,
    data = gspc, n.start = 5133,
    refit.window = "moving", refit.every = length(gspc) - 5134
)

# use "all", "stats", "rmse", "coef" or "coefse" as an argument for which.
print("rmse")
preds <- fpm(garchroll)
print(preds)

# preds1 <- as.data.frame(garchroll, which = "stats")
# print(preds1)

# preds2 <- as.data.frame(garchroll, which = "coefse")
# print(preds2)

png(filename = "C:/Users/nada/OneDrive/Documents/Desktop/model6.png")
plot(garchroll, which = 2)
dev.off()

png(filename = "C:/Users/nada/OneDrive/Documents/Desktop/model7.png")
plot(garchroll, which = 4)
dev.off()
