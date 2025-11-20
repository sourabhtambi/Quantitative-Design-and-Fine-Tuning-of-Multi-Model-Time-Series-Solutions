# Import Dataset
library(quantmod)
library(forecast)
library(fGarch)
library(tseries)
library(ggplot2)

getSymbols("^GSPC", from="2005-01-01", to="2022-01-01")  # S&P 500 index
sp <- GSPC

sp_close <- sp$GSPC.Close
sp_monthly <- to.monthly(sp_close, indexAt = "lastof", OHLC = FALSE)
sp_ts <- ts(as.numeric(sp_monthly), start = c(2005, 1), frequency = 12)


plot(sp_ts, type="l", main="S&P 500 Monthly Close Price",
     ylab="Close Price", xlab="Date")

###ARIMA Model###

par(mfrow=c(1,2))
Acf(sp_ts, main="ACF")
Pacf(sp_ts, main="PACF")
dev.off()

print("ADF Test")
adf.test(sp_ts, alternative = "stationary") # testing for stationarity (optional, as auto.arima will do this)
# This will likely show non-stationarity, which is why auto.arima will use d=1 or d=2


########### Split Data ###########
train <- window(sp_ts, end = c(2020, 12))
test  <- window(sp_ts, start = c(2021, 1))

############# Fit ARIMA on Training Only ###############
fit_train <- auto.arima(train, seasonal=TRUE)

############# Forecast Same Horizon as Test ###############
fcast_train <- forecast(fit_train, h = length(test))
print(fcast_train)

################ Plot Training vs Forecast vs Actual ###########
plot(fcast_train, main="ARIMA: Training Forecast vs Actual Test Data")
lines(test, col="red", lwd=2)
legend("topleft",
       legend = c("Training Forecast", "Actual Test Data"),
       col = c("blue", "red"),
       lwd = 2,
       bty = "n")


======================================================================
======================================================================
## ARCH and GARCH

ret <- diff(log(sp_close))
ret <- na.omit(ret)


ret_monthly <- diff(log(sp_monthly))
ret_monthly<- na.omit(ret_monthly)


# Convert to monthly time series
ret_ts <- ts(as.numeric(ret_monthly),
             start=c(2005,2),
             frequency=12)

=======================================================================
=======================================================================
### --- ACF/PACF of returns, |returns|, returns^2 --- ###
par(mfrow=c(3,2), mar=c(2,2,2,2))
acf(ret_ts, 25, main="ACF(Returns)")
pacf(ret_ts, 25, main="PACF(Returns)")
acf(abs(ret_ts), 25, main="ACF(|Returns|)")
pacf(abs(ret_ts), 25, main="PACF(|Returns|)")
acf(ret_ts^2, 25, main="ACF(Returns^2)")
pacf(ret_ts^2, 25, main="PACF(Returns^2)")

### --- Rolling Historical Volatility (20-day and 40-day) --- ###
n <- length(ret_ts)

hist.vol1 <- sapply(1:(n-20+1), function(i) var(ret_ts[i:(i+19)]))
hist.vol2 <- sapply(1:(n-40+1), function(i) var(ret_ts[i:(i+39)]))

par(mfrow=c(2,1))
plot(hist.vol1, type="l", main="20-Month Historical Volatility")
plot(hist.vol2, type="l", main="40-Month Historical Volatility")

=========================================================================
==========================================================================

### --- Fit ARCH(2) model --- ###
fit1 <- garchFit(~ garch(2,0), data=ret_ts)
fit1.vol <- volatility(fit1)
fit1.resid.st <- residuals(fit1) / fit1.vol

x.pos <- c(seq(1,n,1100), n)

par(mfrow=c(3,2), mar=c(2,2,2,2))
plot(fit1.vol, type="l", xaxt="n")
axis(1, x.pos)
plot(fit1.resid.st, type="l", xaxt="n")
axis(1, x.pos)
acf(fit1.resid.st, 25, main="")
acf(fit1.resid.st, 25, type="partial", main="")
acf(fit1.resid.st^2, 25, main="")
acf(fit1.resid.st^2, 25, type="partial", main="")

===========================================================================
===========================================================================                    

### --- Fit GARCH(1,1) model --- ###
fit2 <- garchFit(~ garch(1,1), data=ret_ts)
fit2.vol <- volatility(fit2)
fit2.resid.st <- residuals(fit2) / fit2.vol

par(mfrow=c(3,2), mar=c(2,2,2,2))
plot(fit2.vol, type="l", xaxt="n")
axis(1, x.pos)
plot(fit2.resid.st, type="l", xaxt="n")
axis(1, x.pos)
acf(fit2.resid.st, 25, main="")
acf(fit2.resid.st, 25, type="partial", main="")
acf(fit2.resid.st^2, 25, main="")
acf(fit2.resid.st^2, 25, type="partial", main="")

===========================================================================
===========================================================================
                    

### --- Fit ARMA(1,1) + GARCH(1,1) --- ###
fit3 <- garchFit(~ arma(1,1) + garch(1,1), data=ret_ts)
fit3.vol <- volatility(fit3)
fit3.resid.st <- residuals(fit3) / fit3.vol

par(mfrow=c(3,2), mar=c(2,2,2,2))
plot(fit3.vol, type="l", xaxt="n")
axis(1, x.pos)
plot(fit3.resid.st, type="l", xaxt="n")
axis(1, x.pos)
acf(fit3.resid.st, 25, main="")
acf(fit3.resid.st, 25, type="partial", main="")
acf(fit3.resid.st^2, 25, main="")
acf(fit3.resid.st^2, 25, type="partial", main="")

### --- Forecast (10-step ahead) --- ###
predict(fit3, n.ahead = 10)

# Volatility Forecasts
v1 <- predict(fit1, n.ahead=10)$standardDeviation
v2 <- predict(fit2, n.ahead=10)$standardDeviation
v3 <- predict(fit3, n.ahead=10)$standardDeviation

# Plot all volatility forecasts
plot(v1, type="l", lwd=2, main="10-Step Ahead Volatility Forecast Comparison",
     ylab="Forecasted Volatility", xlab="Steps")
lines(v2, col="blue", lwd=2)

legend("bottomright", legend=c("ARCH(2)", "GARCH(1,1)"),
       col=c("black","blue"), lwd=2)

===================================================================================
==================================================================================
                    
                    
                    
#################Neural Network ############################

                    
nValid <- 36
train.ts <- window(ret_ts, start = c(2005, 2), end = c(2018, 12))
valid.ts <- window(ret_ts, start = c(2019, 1), end = c(2021, 12))

set.seed(201)
sp.nnetar <- nnetar(train.ts, repeats = 20, p = 11, P = 1, size = 7)
summary(sp.nnetar$model[[1]])

sp.nnetar.pred <- forecast(sp.nnetar, h = nValid)
acc_manual <- accuracy(sp.nnetar.pred, valid.ts)


plot(train.ts,
     ylim = range(c(train.ts, valid.ts, sp.nnetar.pred$mean)),
     ylab = "Log Returns",
     xlab = "Time",
     bty = "l",
     xaxt = "n",
     xlim = c(2005, 2023))

axis(1, at = seq(2005, 2022, 1), labels = seq(2005, 2022, 1))

# Fitted values
lines(sp.nnetar.pred$fitted, col = "blue", lwd = 2)

lines(valid.ts, col = "black", lwd = 1.5)

# Forecasted returns
lines(ts(sp.nnetar.pred$mean,
         start = c(2019, 1),
         frequency = 12),
      col = "red", lwd = 2, lty = 2)

# Vertical boundaries
abline(v = 2019, lwd = 1)
abline(v = 2022, lwd = 1)

# Labels
text(2007, max(train.ts)*0.9, "Training")
text(2020.5, max(train.ts)*0.9, "Validation")
text(2022.5, max(train.ts)*0.9, "Future")

# Arrows
arrows(2005, max(train.ts)*0.8, 2019, max(train.ts)*0.8,
       code = 3, length = 0.1, angle = 30)
arrows(2019, max(train.ts)*0.8, 2022, max(train.ts)*0.8,
       code = 3, length = 0.1, angle = 30)
arrows(2022, max(train.ts)*0.8, 2023, max(train.ts)*0.8,
       code = 3, length = 0.1, angle = 30)


