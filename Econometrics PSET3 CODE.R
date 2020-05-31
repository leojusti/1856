library(quantreg)
install.packages("AER")
library(AER)


x <- lm((educ) ~ (brthord), data=wage2)

summary(x)
x
attach(wage2)

cor.test(educ, brthord, type="pearson")

y <- ivreg(log(wage) ~ educ | brthord)
summary(y)

iv1 <- summary(y, se="boot", bsmethod="mcmb")
iv1

latex.summary.iv(iv1, tranpose=FALSE, caption="IV Regression Summary", digits=2)
