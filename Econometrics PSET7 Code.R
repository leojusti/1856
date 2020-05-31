attach(mathpnl)


x <- mathpnl[!duplicated(mathpnl[c("distid", "year")]),]

ols <- plm(math4 ~ y94+	y95+	y96+	y97+	y98+	lenrol+	rexpp	+lrexpp	+lrexpp_1 + lunch,
           model = "pooling", data = x)
summary(ols)

#rexpp has small positive correlation with dep var
#