function [acc, nmi, ari, f1_macro] = evaluation(y_true, y_pred)
% EVALUATION_STRICT  Evaluate clustering with strict one-to-one mapping
%   This function uses strict Hungarian algorithm matching: one-to-one mapping
%   When number of predicted clusters > number of true classes,
%   unmatched clusters are considered misclassified.
%
%   Inputs:
%     y_true - Nx1 ground-truth labels (numeric/integer)
%     y_pred - Nx1 predicted labels (numeric/integer)
%   Outputs:
%     acc     - accuracy after best label assignment (0..1)
%     nmi     - normalized mutual information (arithmetic mean normalization)
%     ari     - adjusted rand index
%     f1_macro- macro-averaged F1 score

if nargin < 2
    error('Need y_true and y_pred');
end

y_true = y_true(:);
y_pred = y_pred(:);
if numel(y_true) ~= numel(y_pred)
    error('y_true and y_pred must have same length.');
end
N = numel(y_true);
if N == 0
    acc = NaN; nmi = NaN; ari = NaN; f1_macro = NaN;
    return;
end

% --- remap labels to consecutive 1..K indices ---
[~, ~, yt] = unique(y_true);    % yt: 1..K1
[~, ~, yp] = unique(y_pred);    % yp: 1..K2
K1 = max(yt);
K2 = max(yp);

% --- contingency matrix (counts) ---
C = zeros(K1, K2);
for i = 1:K1
    rows = (yt == i);
    for j = 1:K2
        C(i,j) = sum(rows & (yp == j));
    end
end

% --- STRICT ONE-TO-ONE MAPPING: Hungarian algorithm ---
% We match min(K1, K2) pairs. If K2 > K1, some clusters won't be matched.
% Unmatched clusters are considered entirely misclassified.

maxC = max(C(:));
if isempty(maxC)
    maxC = 0;
end
cost = maxC - C;  % smaller cost -> prefer larger C

% Find optimal matching using available method
pairs = [];
try
    % Use MATLAB's matchpairs if available (R2019a+)
    costUnmatched = max(cost(:)) + 1;
    [pairs, ~] = matchpairs(cost, costUnmatched);
catch
    % If matchpairs unavailable, try munkres
    if exist('munkres','file') == 2
        assign = munkres(cost); % assign(i) = j or 0
        rows = find(assign > 0);
        pairs = [rows(:), assign(rows)'];
    else
        % Greedy fallback: match largest counts first
        Ccopy = C;
        pairs = zeros(0,2);
        % Match min(K1, K2) pairs
        for match_idx = 1:min(K1, K2)
            [v, idx] = max(Ccopy(:));
            if v == 0
                break;  % No more positive counts to match
            end
            [r, c] = ind2sub(size(Ccopy), idx);
            pairs = [pairs; r, c];
            % Zero out entire row and column to prevent reuse
            Ccopy(r, :) = 0;
            Ccopy(:, c) = 0;
        end
    end
end

% Build mapping: only matched clusters get a true class assignment
map_pred_to_true = zeros(1, K2);  % 0 means not matched
for t = 1:size(pairs,1)
    r = pairs(t,1);
    c = pairs(t,2);
    map_pred_to_true(c) = r;
end

% --- Calculate ACC: only samples in matched clusters can be correct ---
correct = 0;
for j = 1:K2
    if map_pred_to_true(j) > 0
        % This cluster is matched to a true class
        true_label = map_pred_to_true(j);
        % Count samples in this cluster that have the mapped true label
        correct = correct + C(true_label, j);
    end
    % If map_pred_to_true(j) == 0, all samples in cluster j are wrong
end
acc = correct / N;

% --- For F1 calculation, we need to assign all clusters to some class ---
% We'll use the same mapping, but for unmatched clusters we need to assign
% them to some class (or they'll be treated as a separate class).
% Here we assign unmatched clusters to the true class they have most samples in,
% but note this is only for F1 calculation, not for ACC.
new_pred = zeros(N,1);
for j = 1:K2
    idx = (yp == j);
    if map_pred_to_true(j) > 0
        new_pred(idx) = map_pred_to_true(j);
    else
        % For unmatched clusters in F1 calculation, assign to majority class
        [~, majority_class] = max(C(:, j));
        new_pred(idx) = majority_class;
    end
end

% --- Macro F1 ---
f1s = zeros(K1,1);
for i = 1:K1
    tp = sum((yt == i) & (new_pred == i));
    pred_i = sum(new_pred == i);
    true_i = sum(yt == i);
    
    if pred_i == 0
        prec = 0;
    else
        prec = tp / pred_i;
    end
    
    if true_i == 0
        rec = 0;
    else
        rec = tp / true_i;
    end
    
    if (prec + rec) == 0
        f1s(i) = 0;
    else
        f1s(i) = 2 * prec * rec / (prec + rec);
    end
end
f1_macro = mean(f1s);

% --- Adjusted Rand Index (ARI) ---
% ARI uses the original contingency matrix C (before any mapping)
n = N;
comb2 = @(x) x .* (x - 1) / 2;
sumComb = sum(sum(comb2(C)));
a = sum(comb2(sum(C,2))); % sum over rows
b = sum(comb2(sum(C,1))); % sum over cols
combN = comb2(n);

if combN == 0
    ari = 0;
else
    expectedIndex = a * b / combN;
    maxIndex = 0.5 * (a + b);
    ari = (sumComb - expectedIndex) / (maxIndex - expectedIndex);
end

% --- Normalized Mutual Information (NMI) ---
% NMI also uses the original contingency matrix C
Pij = C / n;
Pi = sum(Pij, 2);
Pj = sum(Pij, 1);

MI = 0;
for i = 1:K1
    for j = 1:K2
        if Pij(i,j) > 0
            MI = MI + Pij(i,j) * log(Pij(i,j) / (Pi(i) * Pj(j)));
        end
    end
end

H_true = - sum( Pi(Pi>0) .* log(Pi(Pi>0)) );
H_pred = - sum( Pj(Pj>0) .* log(Pj(Pj>0)) );

if (H_true + H_pred) == 0
    nmi = 0;
else
    % Arithmetic mean normalization (like sklearn)
    nmi = MI / (0.5 * (H_true + H_pred));
end

end