install.packages("plm")
library(plm)

install.packages("ivpack")
library(ivpack)

install.packages("ggplot2")
library(ggplot2)

library(stargazer)

attach(X2408data)


######### ols

attach(newData)

ols <- plm(nx ~ r,
           model = "pooling", data = newData)
summary(ols)


summary(ols2 <- plm(r ~ fdi, model = "pooling", data = newData))

plot(ols1)
abline(ols1)

mod <- lm(nx ~ r)
abline(mod)

plot(mod)

######### IV

attach(newData)



summary(iv1 <- ivreg(nx ~ r | er))
summary(iv2 <- ivreg(nx ~ r | er + fdi))
summary(iv3 <- ivreg(nx ~ r | er + i))
summary(iv4 <- ivreg(nx ~ r | fdi + i))
summary(iv5 <- ivreg(nx ~ r | er + fdi + i))


summary(iv1,  diagnostics = TRUE)
summary(iv2, diagnostics = TRUE)
summary(iv3, diagnostics = TRUE)
summary(iv4, diagnostics = TRUE)
summary(iv5, diagnostics = TRUE)

stargazer(iv5,diagnostics=TRUE, title="IV Regression Using Exchange Rate as IV", 
          align=TRUE, no.space=TRUE,
          add.lines = list(c(rownames(summ.fit1$diagnostics)[3], 
                             round(summ.fit1$diagnostics[2, "p-value"], 5), 
                             round(summ.fit2$diagnostics[2, "p-value"], 5)), 
                           c(rownames(summ.fit1$diagnostics)[2], 
                             round(summ.fit1$diagnostics[2, "p-value"], 5), 
                             round(summ.fit2$diagnostics[2, "p-value"], 5))), 
          #covariate.labels = c("hi_empunion"),
          dep.var.caption = "Dependant Variable",
          dep.var.labels  = "Net Exports",
          model.names = TRUE, 
          #column.labels = c("OLS"),
          out="2408reg1.tex")

stargazer(iv3,diagnostics=TRUE, title="IV Regression Using Exchange Rate and FDI as IVs", 
          align=TRUE, no.space=TRUE,
          add.lines = list(c(rownames(summ.fit1$diagnostics)[3], 
                             round(summ.fit1$diagnostics[2, "p-value"], 5), 
                             round(summ.fit2$diagnostics[2, "p-value"], 5)), 
                           c(rownames(summ.fit1$diagnostics)[2], 
                             round(summ.fit1$diagnostics[2, "p-value"], 5), 
                             round(summ.fit2$diagnostics[2, "p-value"], 5))), 
          #covariate.labels = c("hi_empunion"),
          dep.var.caption = "Dependant Variable",
          dep.var.labels  = "Net Exports",
          model.names = TRUE, 
          #column.labels = c("OLS"),
          out="2408reg1.tex")

stargazer(iv4,diagnostics=TRUE, title="IV Regression Using Exchange Rate, FDI and Inflation as IVs", 
          align=TRUE, no.space=TRUE,
          add.lines = list(c(rownames(summ.fit1$diagnostics)[3], 
                             round(summ.fit1$diagnostics[2, "p-value"], 5), 
                             round(summ.fit2$diagnostics[2, "p-value"], 5)), 
                           c(rownames(summ.fit1$diagnostics)[2], 
                             round(summ.fit1$diagnostics[2, "p-value"], 5), 
                             round(summ.fit2$diagnostics[2, "p-value"], 5))), 
          #covariate.labels = c("hi_empunion"),
          dep.var.caption = "Dependant Variable",
          dep.var.labels  = "Net Exports",
          model.names = TRUE, 
          #column.labels = c("OLS"),
          out="2408reg1.tex")

stargazer(ols, 
          title="OLS Regression Results", align=TRUE, 
          dep.var.labels=c("Net Exports"), 
          covariate.labels=c("Interest Rate"), 
          omit.stat=c("LL","ser","f"), no.space=TRUE)

stargazer(iv1, diagnostics = TRUE,
          title="IV Regression Using Exchange Rate as IV", align=TRUE, 
          dep.var.labels=c("Net Exports"), 
          covariate.labels=c("Interest Rate"), 
          omit.stat=c("LL","ser","f"), no.space=TRUE)


attach(ivregdata)

iv0 <- ivreg(nx ~ r | er + fdi)
iv0

# m should be pos cor with r
# x should be neg cor with r

attach(Time_series_data)

attach(TS2408total)

ggplot(TS2408total, aes(x = Year, y = Amount, color = Type)) +
  geom_line(aes(linetype = Type)) +
labs(x = "Year",  y = "Percent Change", 
     title = "Interest Rate, Imports and Exports 1990-2015" ,"Imports", "Interest Rate", "Exports") + 
  theme_light()


attach(TSlow)

ggplot(TSlow, aes(x = Year, y = Amount, color = Type)) +
  geom_line(aes(linetype = Type)) +
  labs(x = "Year",  y = "Percent Change", 
       title = "Interest Rate, Imports and Exports 1995-2015 Low Income Country Sample") + 
  theme_light()

attach(TSmed)

ggplot(TSmed, aes(x = Year, y = Amount, color = Type)) +
  geom_line(aes(linetype = Type)) +
  labs(x = "Year",  y = "Percent Change", 
       title = "Interest Rate, Imports and Exports 1995-2015 Medium Income Country Sample") + 
  theme_light()

attach(TShigh)

ggplot(TShigh, aes(x = Year, y = Amount, color = Type)) +
  geom_line(aes(linetype = Type)) +
  labs(x = "Year",  y = "Percent Change", 
       title = "Interest Rate, Imports and Exports 1995-2015 High Income Country Sample") + 
  theme_light()
