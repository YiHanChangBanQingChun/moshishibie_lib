function out = University_colorchange(CMap)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

out=zeros(size(CMap,1),size(CMap,2),3);


[x1,y1]=find(CMap==1);
for i=1:length(x1)
    out(x1(i),y1(i),1)=192;
    out(x1(i),y1(i),2)=192;
    out(x1(i),y1(i),3)=192;
end

[x2,y2]=find(CMap==2);
for i=1:length(x2)
    out(x2(i),y2(i),1)=0;
    out(x2(i),y2(i),2)=255;
    out(x2(i),y2(i),3)=0;
end

[x3,y3]=find(CMap==3);
for i=1:length(x3)
    out(x3(i),y3(i),1)=0;
    out(x3(i),y3(i),2)=255;
    out(x3(i),y3(i),3)=255;
end


[x4,y4]=find(CMap==4);
for i=1:length(x4)
    out(x4(i),y4(i),1)=0;
    out(x4(i),y4(i),2)=128;
    out(x4(i),y4(i),3)=0;
end

[x5,y5]=find(CMap==5);
for i=1:length(x5)
    out(x5(i),y5(i),1)=255;
    out(x5(i),y5(i),2)=0;
    out(x5(i),y5(i),3)=255;
end

[x6,y6]=find(CMap==6);
for i=1:length(x6)
    out(x6(i),y6(i),1)=188;
    out(x6(i),y6(i),2)=95;
    out(x6(i),y6(i),3)=0;
end

[x7,y7]=find(CMap==7);
for i=1:length(x7)
    out(x7(i),y7(i),1)=155;
    out(x7(i),y7(i),2)=52;
    out(x7(i),y7(i),3)=105;
end

[x8,y8]=find(CMap==8);
for i=1:length(x8)
    out(x8(i),y8(i),1)=255;
    out(x8(i),y8(i),2)=0;
    out(x8(i),y8(i),3)=0;
end

[x9,y9]=find(CMap==9);
for i=1:length(x9)
    out(x9(i),y9(i),1)=255;
    out(x9(i),y9(i),2)=255;
    out(x9(i),y9(i),3)=0;
end