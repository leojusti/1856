library(quantreg)
install.packages("AER")
library(AER)

install.packages("stargazer")
library(stargazer)

install.packages("systemfit")
library(systemfit)

install.packages("plm")
library(plm)

install.packages("ivpack")
library(ivpack)

#### iv regression, multi iv

x <- with(mus, ave(hi_empunion==1,
                               FUN=function(x) ifelse(x, cumsum(x), NA)) )

x


attach(mus)


tail(x, n=10)

y <- lm(ldrugexp ~ hi_empunion + totchr + age #first stage reg
      + female + blhisp + linc)
summary(y, robust = TRUE)
summary(y)
  

z <- ivreg(ldrugexp ~ hi_empunion | ssiratio ) #2nd stage reg

summ.fit1<-summary(z, diagnostics = TRUE, robust = TRUE)
summ.fit2<-summary(z2, diagnostics = TRUE, robust = TRUE)

summ.fit1
summ.fit2

#ssiratio must have a causel effect on hi_emp, 
#ssiratio affects outcome ldrugexp only through hi_emp
#ssiratio does not share common causes with outcome ldrug 



z2 <- ivreg(ldrugexp ~ hi_empunion| multlc + ssiratio)
summary(z2, diagnostics = TRUE, robust = TRUE)

#The syntax for ivreg is as follows: ivreg(Y ~ X + W | W + Z, ... ), 
#where X is endogenous variable(s), 
#Z is instrument(s), and W is exogenous controls (not instruments).

d0 = robust.se(y)
d0

d1 = robust.se(z)
d1

d2 = robust.se(z2)
d2


###############
z3 <- ivreg(ldrugexp ~ hi_empunion + totchr + age 
           + female + blhisp + linc | linc + blhisp + female + age + totchr + ssiratio ) 

summary(z3, diagnostics = TRUE, robust = TRUE)

z4 <- ivreg(ldrugexp ~ totchr + age 
             + female + blhisp + linc |linc + blhisp + female + age + totchr + hi_empunion + ssiratio )
summary(z4, robust = TRUE)

#######

out <- capture.output(summary(y))

cat("My title", out, file="summary_y.txt", sep="n", append=TRUE)

#educ ~ brthord
stargazer(y, title="OLS Regression Results", 
          align=TRUE, no.space=TRUE, 
          #covariate.labels = c("hi_empunion","totchr","age", "female", "blhisp","linc"),
          dep.var.caption = "Dependant Variable",
          dep.var.labels  = "ldrugexp",
          model.names = FALSE, 
          #column.labels = c("OLS"),
          out="olsreg.tex")


stargazer(z,diagnostics=TRUE, title="IV Regression Using ssiratio as IV", 
          align=TRUE, no.space=TRUE,
          add.lines = list(c(rownames(summ.fit1$diagnostics)[1], 
                             round(summ.fit1$diagnostics[1, "p-value"], 2), 
                             round(summ.fit2$diagnostics[1, "p-value"], 2)), 
                           c(rownames(summ.fit1$diagnostics)[2], 
                             round(summ.fit1$diagnostics[2, "p-value"], 2), 
                             round(summ.fit2$diagnostics[2, "p-value"], 2))), 
          #covariate.labels = c("hi_empunion"),
          dep.var.caption = "Dependant Variable",
          dep.var.labels  = "ldrugexp",
          model.names = FALSE, 
          #column.labels = c("OLS"),
          out="olsreg.tex")

stargazer(z2, title="IV Regression Using ssiratio and multlc as IVs", 
          align=TRUE, no.space=TRUE, 
          #covariate.labels = c("hi_empunion"),
          dep.var.caption = "Dependant Variable",
          dep.var.labels  = "ldrugexp",
          model.names = FALSE, 
          #column.labels = c("OLS"),
          out="olsreg.tex")

hausman.systemfit(y,z2)
#phtest(y,z)
