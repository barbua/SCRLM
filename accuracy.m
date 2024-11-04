function acc= accuracy(y_true, y_pred)
table = crosstab(y_true, y_pred);
M= matchpairs(-table,1e-14);
a=M(:,1).';
b=M(:,2).';
acc=sum(sum(table(sub2ind(size(table),a,b))))/sum(sum(table));
end