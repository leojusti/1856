cls
*Import Dataset
import delimited "/Users/justinleo/Desktop/ECO2404.2/data-olley-pakes.csv", clear 

*Question 1:

replace output = . if output==0
replace capital = . if capital==0
replace labor = . if labor==0
replace inv = . if inv==0
replace age = . if age==0

gen not_missing = output != .
egen years_exist = sum(not_missing), by(firm)

*Transforming the variables
gen lo=log(output)
gen ll=log(labor)
gen lc=log(capital)
gen li=log(inv)

*1a - *Unbalanced regression
reg lo ll lc



*1b - *Balanced regression
mvdecode _all,mv(0=.)
gen missing=output!=.
egen all=sum(missing),by(firm)

reg lo ll lc  if all==6


*esttab using question1b.tex, se ar2

*Question 2:
egen  i=group(firm)
xtset i year, yearly

xtreg lo ll lc , fe


*esttab fixed using question2.tex, se ar2

*Question 3:
xtabond2 lo L1.lo ll lc, gmm(L1.lo L1.ll L1.lc, collapse) iv(L2.ll L2.lc, equation(level))

//xtabond2 lo l.lo lc ll, gmmstyle(ll lc)

*esttab gmm using question3.tex, se ar2

*Question 4A:

forvalues i=1/2 {
gen phi`i'_2=li^`i'*lc^(3-`i')
}
rename phi2_2 phi2_1

gen phi2_2 = li^2*lc^2
gen phi1_1 = li*lc^1
gen loginv2 = li^2
gen logcapital2 = lc^2


regress lo ll lc li logcapital2 loginv2 phi*, vce(cluster firm)

predict double lny_hat if e(sample), xb
scalar b_loglabor = _b[ll]
generate double phi_hat = lny_hat - ll*b_loglabor 

*esttab using question4aNEW.tex, se ar2

*Question 4B

// gen logoutput=log(output)
// gen loglabor=log(labor)
// gen logcapital=log(capital)
// gen loginv=log(inv)


prodest lo, state(lc) free(ll) proxy(li) poly(4) method(op)

scalar bll = _b[ll]
scalar blc = _b[lc] 

*esttab using question4b.tex, se ar2

*Question 4D

// gen logoutput=log(output)
// gen loglabor=log(labor)
// gen logcapital=log(capital)
// gen loginv=log(inv)

* Firm's Productivity
generate prodg = lo - ll*bll - lc*blc
generate prodgrowthfirm = exp(prodg)

*Total output by year
egen sumoutput_byyear=total(output), by (year)

*Firm level market shares by year
generate firmmkshr = output/sumoutput_byyear

* Mean productivity by year
egen mean_prod = mean(prodgrowthfirm), by(year)

*generate average market share by year
bysort year: egen avgmkshr = mean(firmmkshr)

* Generate the summation term of equation 16 (Olley and Pakes, 1290)
generate pmktshr=prodgrowthfirm*firmmkshr

* Generate the industry productivity by year
bysort year: egen total_share = total(pmktshr)

*Unweighted average prod by year
bysort year: egen total_prod = mean(prodgrowthfirm)

*delta Share
generate deltaShare=firmmkshr-avgmkshr

*delta Productivity
generate deltaProd=prodgrowthfirm-total_prod

*deltaSP
generate deltasp=deltaShare*deltaProd

*sum deltaSP
bysort year: egen sumdeltasp=total(deltasp)

*correlation p,k
bysort year: egen correlation = corr(prodgrowthfirm lc)

sort firm year
order firm year  total_prod total_share

sort firm year
order firm year  total_share total_prod


*4e

gen logoutput=log(output)
gen loglabor=log(labor)
gen logcapital=log(capital)
gen loginv=log(inv)

gen exit = 1*(output == .)
rename logcapital lk
// rename loglabor ll

replace logoutput = 0 if logoutput == .
replace lk = 0 if lk == .
replace ll = 0 if ll == .
replace loginv = 0 if loginv == .

opreg logoutput, exit(exit) state(lk) free(ll) proxy(loginv)

*esttab using question4e4.tex, keep(ll lk) se ar2
