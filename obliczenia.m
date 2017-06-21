clear all;

n=[0;10;20;25;50;75;100;150;200;500;1000;2500;5000;10000;25000;40000;50000;55000;70000;80000;100000];
t=[0;243;244;244;295;295;296;295;344;411;461;551;776;917;1109;1397;1597;1841;2331;2442;2556];

x=0:10:100000;
y1=400*log(x*0.0005) +700;
y2=x*0.1+10;

plot(n,t,'*',x,y1,'-' ,x,y2,'-');
xlabel("n");
ylabel("t [ns]");
title("Czas wyszukiwania pesymistycznego w drzewie AVL");
legend("wyszukiwanie w drzewie AVL","y1=log(n)","y2=n")