
%B1SUP
R_B1SUP_1 = zeros(9,9,8);
for i = 1:12
    for j = 2:4:14
        s = RB1SUP(i,j) + 1;
        a = RB1SUP(i,j+1);
        s_ = RB1SUP(i,j+2) + 1;
        R_B1SUP_1(s,s_,a) = 1;
    end
end
R_B1SUP_1(8,7,5) = 1;  
R_B1SUP_1(9,8,3) = 1; 
R_B1SUP_1(9,9,4) = 1; 
R_B1SUP = R_B1SUP_1;

%R_B2SUP
R_B2SUP_2 = zeros(3,3,8);
for i = 1:4
    for j = 2:4:14
        s = RB2SUP(i,j) + 1;
        a = RB2SUP(i,j+1);
        s_ = RB2SUP(i,j+2) + 1;
        R_B2SUP_2(s,s_,a) = 1;
    end
end
R_B2SUP_2(3,3,6) = 1;
R_B2SUP_2(3,3,8) = 1;
R_B2SUP = R_B2SUP_2;