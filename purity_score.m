function acc=purity_score(x,y)
table = crosstab(x,y);
[max_val,~]=max(table);
acc=sum(max_val)/sum(sum(table));
end