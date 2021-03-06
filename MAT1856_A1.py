#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:38:17 2020

@author: justinleo
"""

import numpy as np 
import math





 
mar20 = [100.35137,
        100.365479,
        100.377808, 
        100.381918,
100.386027,
100.390137,
100.394247,
100.406575,
100.410685, 
100.414795]
 

                                #r1/6s
                                
print ("R2")
new_list = []
    
def r_calc1(t, N=100.75):
    for i in mar20:
        #x = (-math.log10(i/N))/(t)
        x = (-np.log(i/N))/(t)
        new_list.append(x)
    return new_list


index = [0,1,2,3,4,5,6,7,8,9]


for i in index:
    print(r_calc1((2/12))[i])
    
sep20 = [99.5106849,
99.5327397,
99.5389041,
99.5309589,
99.5430137,
99.5450685,
99.5471233,
99.5432877,
99.5553425,
99.5773973]

#copy paste output for rs in an array to loop over for next function


r2= [0.02378684076237743,
0.022943324131251,
0.022206323146574898,
0.021960656343827746,
0.021715059368638972,
0.021469412679189587,
0.02122377604637848,
0.020487045993619907,
0.020241449574234394,
0.019995863207370273]



print("")

print ("R8")

                                #r8/12s
new_list2 = []

def r_calc2(t, c):
    for f,b in zip(sep20,r2):
        a = f - (c/2)*math.exp(-b*t)
        b = a/(100+(c/2))
        x2 = (np.log(b))*-12/8
        new_list2.append(x2)
    return new_list2


for i in index:
    print(r_calc2((2/12),0.75)[i])



r8 = [0.018613109020627953,
0.01828023925735205,
0.01818768621957243,
0.018308103615784128,
0.01812598734808418,
0.01809513889215051,
0.018064291070621952,
0.018122999755995576,
0.017940906015453072,
0.01760762263459859,]

mar21 = [99.14068493,
99.18273973,
99.20890411,
99.2009589,
99.1830137,
99.18506849,
99.14712329,
99.17328767,
99.17534247,
99.20739726]



print("")
print ("R14")

new_list3 = []
    
                                            #r14/12
def r_calc3(t1,t2,c=0.75):
    for f,b,k in zip(mar21,r2, r8):
        a = f - (c/2)*math.exp(-b*t1) - (c/2)*math.exp(-k*t2)
        b = a/(100+(c/2))
        x3 = (np.log(b))*-12/14
        new_list3.append(x3)
    return new_list3

for i in index:
    print(r_calc3((2/12),(8/12))[i])


print("")

sep21 = [98.66068493,
98.70273973,
98.74890411,
98.7209589,
98.7230137,
98.69506849,
98.69712329,
98.65328767,
98.68534247,
98.69739726]



r14 = [0.01706139280359239,
0.016696301134510417,
0.016469106778398256,
0.01653814742858627,
0.0166949132602123,
0.016677221192766014,
0.017007880265569902,
0.01678027794729192,
0.0167629093626112,
0.01648466996750252]

new_list4 = []

print ("R20")
    
                                        #r20/12
def r_calc4(c,t1=1/6, t2=8/12, t3=14/12):
    for f,b,k,j in zip(mar21,r2, r8, r14):
        a = f - (c/2)*math.exp(-b*t1) - (c/2)*math.exp(-k*t2) - (c/2)*math.exp(-j*t3)
        b = a/(100+(c/2))
        x4 = (np.log(b))*-12/20
        new_list4.append(x4)
    return new_list4

for i in index:
    print(r_calc4(0.75)[i])



print("")

mar22 = [97.73712329,
97.79849315,
97.83260274,
97.8239726,
97.81534247,
97.77671233,
97.78808219,
97.75219178,
97.76356164,
97.79493151]


r20 = [0.014188766729815274,
0.013933202561457974,
0.013774166512179397,
0.013822494967311019,
0.013932231049449262,
0.013919846602236819,
0.014151307953199643,
0.013991986330405026,
0.013979828321128462,
0.013785060744552349]

new_list5 = []
print ("R26")

#r26/12
def r_calc5(c,t1=1/6, t2=8/12, t3=14/12, t4 = 20/12 ):
    for f,b,k,j,l in zip(mar21,r2, r8, r14,r20):
        a = f - (c/2)*math.exp(-b*t1) - (c/2)*math.exp(-k*t2) - (c/2)*math.exp(-j*t3) - (c/2)*math.exp(-l*t4)
        b = a/(100+(c/2))
        x5 = (np.log(b))*-12/26
        new_list5.append(x5)
    return new_list5

for i in index:
    print(r_calc5((0.5))[i])
    

print("")

r26 = [0.009744848112095094,
0.009548545752798605,
0.00942641158220949,
0.009463516060702847,
0.009547620811820536,
0.009538082473794868,
0.009715643910526707,
0.009593349305963901,
0.009583926175783979,
0.009434330519379416]

jun22 = [102.7635616,
102.8310959,
102.8836986,
102.8612329,
102.8587671,
102.8063014,
102.8138356,
102.7764384,
102.7939726,
102.8415068]





print("")
print ("R5")

                                    ### r5
for i,j in zip(r2,r8):
    A = j - i
    B = A/6
    #print(B)
    
print("")

dif1 = [-0.0008734246448335684,-0.0007879233293753644,-0.0006801728149970649,-0.0006190456072980496,-0.0006083502538960483,-0.0005724363082870489,-0.0005365239189543455,-0.0004036089725688458,-0.0003929099142098256,-0.00040740986926400313  ]

                                    

for i,j in zip(r2,dif1):
    n = i+3*j
    print(n)
    
print("")
print ("R11")

                                    ### r11

for i,j in zip(r8,r14):
    A = j - i
    B = A/6
    n = i+3*B
    print(n)
    
print("")
print ("R17")

                                    ### r17

for i,j in zip(r14,r20):
    A = j - i
    B = A/6
    n = i+3*B
    print(n)
    
print("")
print ("R23")

                                    ### r23
    
for i,j in zip(r20,r26):
    A = j - i
    B = A/6
    n = i+3*B
    print(n)
    
    new_list6 = []
    
r5 = [0.021166566827876723,0.020579554143124908,0.020165804701583703,0.020103519521933597,0.019890008606950827,0.01975210375432844,0.019614204289515443,0.01927621907591337,0.019062719831604916,0.018773633599578263]
r11 = [0.01783725091211017,0.017488270195931233,0.017328396498985342,0.0174231255221852,0.01741045030414824,0.01738618004245826,0.017536085668095926,0.017451638851643747,0.017351907689032137,0.017046146301050554]
r17 = [0.015625079766703832,0.015314751847984195,0.015121636645288827,0.015180321197948644,0.015313572154830781,0.015298533897501417,0.015579594109384773,0.015386132138848475,0.01537136884186983,0.015134865356027434]
r23 = [0.011966807420955183,0.01174087415712829,0.011600289047194443,0.011643005514006932,0.011739925930634899,0.011728964538015843,0.011933475931863176,0.011792667818184464,0.01178187724845622,0.011609695631965882]

print("")
print ("R29")
    
    #############
                                    #r29/12
                                    
                                    
def r_calc6(t1=5/12, t2=13/12, t3=17/12, t4 = 23/12 , c=2.75):
    for f,b,k,j,l in zip(jun22,r5, r11, r17,r23):
        a = f - (c/2)*math.exp(-b*t1) - (c/2)*math.exp(-k*t2) - (c/2)*math.exp(-j*t3) - (c/2)*math.exp(-l*t4)
        b = a/(100+(c/2))
        x6 = (np.log(b))*-12/29
        new_list6.append(x6)
    return new_list6

for i in index:
    print(r_calc6((2.75))[i])
    
print("")   

    
r29 = [0.016429053048032636,
0.016158365689221986,
0.01594552675101229,
0.01604025704569931,
0.016051903301769088,
0.01627707743420702,
0.016241747826072333,
0.016409312491215196,
0.016338961189710486,
0.016147268656159453]

print ("R32")


                                    #r32 extrapolation assumption made!!
for i,j in zip(r26,r29):
    A = j - i
    B = A/6
    n = i+3*B
    print(n)
    
print("")
    
mar23= [100.8949315,
101.009726,
101.0841096,
101.0589041,
101.0536986,
100.9684932,
100.9332877,
100.9076712,
100.9524658,
101.0272603]

r32 = [0.013086950580063865,
0.012853455721010296,
0.01268596916661089,
0.01275188655320108,
0.012799762056794811,
0.012907579954000943,
0.012978695868299521,
0.013001330898589548,
0.012961443682747233,
0.012790799587769434]


    
    #############
    
print ("R38")
    
                                        #r38/12
    
new_list7 = []

def r_calc7(t1=2/12, t2=8/12, t3=14/12, t4 = 20/12, t5 = 26/12, t6 = 32/12, c=1.75):
    for f,b,k,j,l,u,v in zip(mar23,r2, r8, r14,r20,r26,r32):
        a = f - (c/2)*math.exp(-b*t1) - (c/2)*math.exp(-k*t2) - (c/2)*math.exp(-j*t3) - (c/2)*math.exp(-l*t4) - (c/2)*math.exp(-u*t5) - (c/2)*math.exp(-v*t6) 
        b = a/(100+(c/2))
        x7 = (np.log(b))*-12/38
        new_list7.append(x7)
    return new_list7

for i in index:
    print(r_calc7((1.75))[i])
    

    
print("")
    

r38 = [0.01637544080549326,
0.01600722699150791,
0.01576958536912879,
0.015852334863427756,
0.015869106076103,
0.016150338856598462,
0.016263814351072735,
0.01635381052354068,
0.016208172069484376,
0.015967633368736585]

print ("R35")
    
                                        #r35
for i,j in zip(r29,r38):
    A = j - i
    B = A/6
    n = i+3*B
    print(n)
    
print("")

r35 = [0.016402246926762946,
0.016082796340364948,
0.01585755606007054,
0.01594629595456353,
0.015960504688936044,
0.01621370814540274,
0.016252781088572532,
0.016381561507377938,
0.016273566629597433,
0.01605745101244802]



jun23 = [99.60739726,
99.72150685,
99.79383562,
99.75794521,
99.77205479,
99.69616438,
99.69027397,
99.61260274,
99.66671233,
99.74082192]

print("")
print ("R41")


                                    #r41/12

new_list8 = []
                            
def r_calc8(t1=5/12, t2=11/12, t3=17/12, t4 = 23/12, t5 = 29/12, t6 = 35/12, c=1.5):
    for f,b,k,j,l,u,v in zip(jun23,r5, r11, r17,r23,r29,r35):
        a = f - (c/2)*math.exp(-b*t1) - (c/2)*math.exp(-k*t2) - (c/2)*math.exp(-j*t3) - (c/2)*math.exp(-l*t4) - (c/2)*math.exp(-u*t5) - (c/2)*math.exp(-v*t6) 
        b = a/(100+(c/2))
        x8 = (np.log(b))*-12/41
        new_list8.append(x8)
    return new_list8

for i in index:
    print(r_calc8((1.5))[i])
    
r41 = [0.016459115160002147,
0.016116803501765205,
0.01590039403429117,
0.016009063705913578,
0.015965518646406354,
0.01619621661557205,
0.016212612934051254,
0.016452013525822387,
0.01628786721338068,
0.016065855483441895]

print("")
print ("R50")

                                    #r50/12

new_list9 = []
                            
def r_calc9(t1=2/12, t2=8/12, t3=14/12, t4 = 20/12, t5 = 26/12, t6 = 32/12, t7 = 38/12, c=2.25):
    for f,b,k,j,l,u,v,w in zip(jun23,r2, r8, r14,r20,r26,r32,r38):
        a = f - (c/2)*math.exp(-b*t1) - (c/2)*math.exp(-k*t2) - (c/2)*math.exp(-j*t3) - (c/2)*math.exp(-l*t4) - (c/2)*math.exp(-u*t5) - (c/2)*math.exp(-v*t6) - (c/2)*math.exp(-w*t7)
        b = a/(100+(c/2))
        x9 = (np.log(b))*-12/50
        new_list9.append(x9)
    return new_list9

for i in index:
    print(r_calc9((2.25))[i])
    
r50 = [0.022768782259367147,
0.022485800071990577,
0.022307794217920493,
0.022400675434587708,
0.022363727071921257,
0.02255999305855815,
0.02257209699056581,
0.02278054982792815,
0.022643048855146373,
0.022458238038332273]

print("")
print ("R44")

    #                               r44
for i,j in zip(r50,r41):
    A = j - i
    B = A/6
    n = i+3*B
    print(n)
    
print("")

r44= [0.019613948709684647,
0.01930130178687789,
0.01910409412610583,
0.019204869570250645,
0.019164622859163805,
0.019378104837065098,
0.019392354962308532,
0.01961628167687527,
0.019465458034263527,
0.019262046760887084]

jun24 = [103.7423288,
104.0791781,
104.229726,
104.1265753,
104.0734247,
103.910274,
103.9671233,
103.8976712,
103.9945205,
104.1013699]

print ("R47")

###                             r47
for i,j in zip(r50,r44):
    A = j - i
    B = A/6
    n = i+3*B
    print(n)
    
print("")

r47 = [0.021191544562974514,
0.020893723443135864,
0.02070611107533631,
0.020802937640103568,
0.020764338249388586,
0.020969210532781543,
0.02098238573902826,
0.021198570149782776,
0.021054405922102448,
0.020860292928353882]

print ("R53")


#                           r53/12
new_list10 = []

def r_calc10(t1=5/12, t2=11/12, t3=17/12, t4 = 23/12, t5 = 29/12, t6 = 35/12, t7 = 41/12, t8 = 47/12, c=2.5):
    for f,b,k,j,l,u,v,w,q in zip(jun24,r5, r11, r17,r23,r29,r35,r41,r47):
        a = f - (c/2)*math.exp(-b*t1) - (c/2)*math.exp(-k*t2) - (c/2)*math.exp(-j*t3) - (c/2)*math.exp(-l*t4) - (c/2)*math.exp(-u*t5) - (c/2)*math.exp(-v*t6) - (c/2)*math.exp(-w*t7) - (c/2)*math.exp(-q*t8)
        b = a/(100+(c/2))
        x10 = (np.log(b))*-12/53
        new_list10.append(x10)
    return new_list10

for i in index:
    print(r_calc10((2.5))[i])
    
sep24 = [99.22136986,
99.45547945,
99.80780822,
99.63191781,
99.7760274,
99.52013699,
99.56424658,
99.60657534,
99.54068493,
99.65479452]

r53 = [0.016442639335998748,
0.015652831630260974,
0.015304800469473037,
0.015547937122548105,
0.015676374367471872,
0.01606073689401779,
0.015922186467431283,
0.0160862076941217,
0.01586026163313548,
0.015616353178340052]
    


print("")
print ("R56")
    
    ##### r56/12
    
new_list11 = []

def r_calc11(t1=2/12, t2=8/12, t3=14/12, t4 = 20/12, t5 = 26/12, t6 = 32/12, t7 = 38/12, t8 = 44/12, t9= 50/12, c=1.5):
    for f,b,k,j,l,u,v,w,q,d in zip(sep24,r2, r8, r14,r20,r26,r32,r38,r44,r50):
        a = f - (c/2)*math.exp(-b*t1) - (c/2)*math.exp(-k*t2) - (c/2)*math.exp(-j*t3) - (c/2)*math.exp(-l*t4) - (c/2)*math.exp(-u*t5) - (c/2)*math.exp(-v*t6) - (c/2)*math.exp(-w*t7) - (c/2)*math.exp(-q*t8) - (c/2)*math.exp(-d*t9)
        b = a/(100+(c/2))
        x11 = (np.log(b))*-12/56
        new_list11.append(x11)
    return new_list11

for i in index:
    print(r_calc11((1.5))[i])


mar25 = [98.44890411,
98.64232877,
98.80260274,
98.7060274,
98.70945205,
98.53287671,
98.48630137,
98.48657534,
98.59,
98.72342466]

r56 = [
0.017755542888114175,
0.017226586929818852,
0.01642360772676881,
0.016826035249561715,
0.016494821837127045,
0.017079700479839952,
0.01697575159160471,
0.016878197998704915,
0.0170335295978429,
0.016777965013465762]


print("")
print ("R62")


#                           r62/12
new_list12 = []

def r_calc12(t1=2/12, t2=8/12, t3=14/12, t4 = 20/12, t5 = 26/12, t6 = 32/12, t7 = 38/12, t8 = 44/12, t9= 50/12, t10 = 56/12, c=1.25):
    for f,b,k,j,l,u,v,w,q,s,z in zip(mar25,r2, r8, r14,r20,r26,r32,r38,r44,r50,r56):
        a = f - (c/2)*math.exp(-b*t1) - (c/2)*math.exp(-k*t2) - (c/2)*math.exp(-j*t3) - (c/2)*math.exp(-l*t4) - (c/2)*math.exp(-u*t5) - (c/2)*math.exp(-v*t6) - (c/2)*math.exp(-w*t7) - (c/2)*math.exp(-q*t8) - (c/2)*math.exp(-s*t9) - (c/2)*math.exp(-z*t10)
        b = a/(100+(c/2))
        x12 = (np.log(b))*-12/62
        new_list12.append(x12)
    return new_list12

for i in index:
    print(r_calc12((1.25))[i])
    
    
r62 = [0.016363132067264634,
0.01596993945516777,
0.01564546259963677,
0.015843219918542685,
0.015837918388481267,
0.016200667796538844,
0.016296813174844608,
0.0162964685729937,
0.01608199119248498,
0.015810015960486055]
    
print("")
print("R6")
    
    
    ##### r6, 
    
for i,j in zip(r2,r8):
    A = (1/3)*i + (2/3)*j

    print(A)
    
print("")

r6 = [0.020337686267877776,
0.01983460088198503,
0.019527231861906587,
0.019525621191798667,
0.01932234468826911,
0.019219896821163533,
0.019117452729207462,
0.01891101516853702,
0.018707753868380177,
0.018403702825522485]

print("")
print("R12")
    
    
    ##### r12, 
    
for i,j in zip(r8,r14):
    A = (1/3)*i + (2/3)*j

    print(A)
    
print("")

print("R18")
    
    
    ##### r18, 
    
for i,j in zip(r14,r20):
    A = (1/3)*i + (2/3)*j

    print(A)
    
print("")

print("R24")
    
    
    ##### r24, 
    
for i,j in zip(r20,r26):
    A = (1/3)*i + (2/3)*j

    print(A)
    
    
print("")
print("R30")


    ##### r30, 
    
for i,j in zip(r29,r38):
    A = (9/10)*i + (1/10)*j

    print(A)
    

print("")
print("R36")


    ##### r36, 
    
for i,j in zip(r29,r38):
    A = (2/10)*i + (8/10)*j

    print(A)
    
print("")
print("R42")


    ##### r42, 
    
for i,j in zip(r41,r44):
    A = (2/3)*i + (1/3)*j

    print(A)
    
print("")
print("R48")


    ##### r48, 
    
for i,j in zip(r44,r50):
    A = (1/3)*i + (2/3)*j

    print(A)
    
print("")
print("R54")


    ##### r54, 
    
for i,j in zip(r53,r56):
    A = (2/3)*i + (1/3)*j

    print(A)
    
print("")
print("R60")


    ##### r60, 
    
for i,j in zip(r56,r62):
    A = (1/3)*i + (2/3)*j

    print(A)
    


print("")

print("forward rates")


R12=[
0.017578632,
0.017224281,
0.017041967,
0.017128133,
0.017171938,
0.01714986,
0.017360017,
0.017227852,
0.017155575,
0.016858988]

R24=[
0.011226154,
0.011010098,
0.010875663,
0.010916509,
0.011009158,
0.010998671,
0.011194199,
0.011059562,
0.011049227,
0.010884574]



R36=[
0.016386163,
0.016037455,
0.015804774,
0.015889919,
0.015905666,
0.016175687,
0.016259401,
0.016364911,
0.01623433,
0.01600356]



R48= [
0.021717171,
0.021424301,
0.021239894,
0.021335407,
0.021297359,
0.021499364,
0.021512183,
0.021725794,
0.021583852,
0.021392841]



R60= [
0.016827269007547815,
0.016388821946718132,
0.015904844308680784,
0.01617082502888236,
0.016056886204696523,
0.016493678690972548,
0.01652312598043131,
0.016490378381564103,
0.016399170660937618,
0.016132665644812624]

print("")


def forward():
    for j,k,m,n,o in zip(R12,R24,R36,R48,R60):
        For12 = ((1+k)**2)/(1+j) - 1
        For13 = ((1+m)**3)/(1+j)**(1/2) - 1
        For14 = ((1+n)**4)/(1+j)**(1/3) - 1
        For15 = ((1+o)**5)/(1+j)**(1/4) - 1
        
        
        
        print(For12)
        print(For13)
        print(For14)
        print(For15)

        
        
        
    return For12, For13, For14, For15


#print(forward())




        
















