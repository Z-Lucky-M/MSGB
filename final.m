datamatrix = table2array(data);
true_labels = datamatrix(:,1);  % 第一列：真实标签（可选）
X = datamatrix(:,2:3);          % 第二、三列：样本特征
X =zscore(X);
[N, D] = size(X);
fprintf('数据加载完成：样本数=%d，特征维度=%d\n', N, D);
if ~isempty(true_labels)
    fprintf('真实标签类别数：%d\n', length(unique(true_labels)));
end

%% ===================== 2. 检测全局密度峰（Mean-Shift） =====================
h = 0.2; % 带宽（控制密度峰检测的全局感知范围）
opts = struct();
opts.maxSeeds = 200;     % 最大种子数
opts.tau = 3.0;          % 密度阈值
opts.epsFactor = 1e-3;   % 收敛精度
opts.maxIter = 300;      % 最大迭代次数
opts.mergeTolFactor = 0.2; % 密度峰合并阈值

fprintf('\n=== 检测全局密度峰 ===\n');
modes = meanShiftClusterModes_kdtree_annotated(X, h, opts);
num_modes = size(modes, 1);
fprintf('检测到全局密度峰数量：%d\n', num_modes);

figure('Position', [100, 100, 1200, 500]);
subplot(1, 2, 1);
scatter(X(:,1), X(:,2), 20, 'k', 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
scatter(modes(:,1), modes(:,2), 100, 'r', 'o', 'LineWidth', 2);
title(sprintf('Mean-shift 检测到的密度峰 (K=%d)', size(modes,1)));
xlabel('X1'); ylabel('X2');
legend('数据点', '密度峰', 'Location', 'best');
grid on;
%% ===================== 3. 构建全局连通的粒球 =====================
fprintf('\n=== 构建自适应半径粒球 ===\n');
params = struct();
params.radiusType = 'adaptive';    % 自适应半径
params.radiusPrctile = 95;         % 半径分位数
params.minSize = 2;                % 最小粒球样本数
params.useMeanShiftSplitting = true; % 基于密度峰拆分粒球

[finalBalls, Centers_final, Radii_final, MeanDensity_final] = ...
    balls_from_modes_kmeans5_adaptive(X, modes, h, params);

numBalls = length(finalBalls);
fprintf('构建粒球总数：%d\n', numBalls);
ball_sizes = cellfun(@length, finalBalls);
fprintf('粒球样本数分布： %s\n', num2str(ball_sizes));

%% ------------------ 第三步：可视化粒球 ------------------
subplot(1, 2, 2);
scatter(X(:,1), X(:,2), 20, 'k', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;

colors = lines(max(1,numBalls));
szs = cellfun(@length, finalBalls);
maxsz = max(szs); if maxsz==0, maxsz=1; end

for i = 1:numBalls
    center = Centers_final{i};
    radius = Radii_final(i);
    ballPoints = finalBalls{i};

    alpha = min(0.7, 0.2 + 0.5 * length(ballPoints) / maxsz);

    theta = linspace(0, 2*pi, 100);
    x_circle = center(1) + radius * cos(theta);
    y_circle = center(2) + radius * sin(theta);
    fill(x_circle, y_circle, colors(mod(i-1,size(colors,1))+1,:), 'FaceAlpha', alpha, ...
        'EdgeColor', colors(mod(i-1,size(colors,1))+1,:), 'LineWidth', 2, 'EdgeAlpha', 0.8);

    plot(center(1), center(2), 'o', 'Color', colors(mod(i-1,size(colors,1))+1,:), ...
        'MarkerFaceColor', colors(mod(i-1,size(colors,1))+1,:), 'MarkerSize', 10, 'LineWidth', 2);

    if ~isempty(ballPoints)
        scatter(X(ballPoints,1), X(ballPoints,2), 30, ...
            'MarkerFaceColor', colors(mod(i-1,size(colors,1))+1,:), 'MarkerEdgeColor', colors(mod(i-1,size(colors,1))+1,:), ...
            'MarkerFaceAlpha', 0.4, 'MarkerEdgeAlpha', 0.6);
    end

    text(center(1), center(2), sprintf('B%d\nn=%d', i, length(ballPoints)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 10, 'FontWeight', 'bold', 'Color', 'white');
end

title(sprintf('几何粒球可视化 (共 %d 个粒球)', numBalls));
xlabel('X1'); ylabel('X2');
grid on;

h1 = plot(NaN, NaN, 'o', 'Color', 'k', 'MarkerFaceColor', 'k', 'MarkerSize', 8);
h2 = plot(NaN, NaN, '-', 'Color', colors(1,:), 'LineWidth', 2);
h3 = plot(NaN, NaN, 'o', 'Color', colors(1,:), 'MarkerFaceColor', colors(1,:), 'MarkerSize', 10);
legend([h1, h2, h3], {'数据点', '粒球边界', '粒球中心'}, 'Location', 'best');

%% ===================== 4. 预处理：生成全局连通的相似度矩阵 =====================
% 4.1 排除异常粒球（仅含1个样本的粒球）
GBs_final_daxiao = cellfun(@length, finalBalls);
normal_GB_biaoqiao = find(GBs_final_daxiao ~= 1); % 正常粒球索引（相对于全部粒球）
outlier_GB_biaoqiao = find(GBs_final_daxiao == 1); % 异常粒球索引
num_normal_balls = length(normal_GB_biaoqiao);
fprintf('\n=== 粒球预处理 ===\n');
fprintf('正常粒球数：%d，异常粒球数：%d\n', num_normal_balls, length(outlier_GB_biaoqiao));

% 4.2 计算粒球间几何特征
normal_Centers = Centers_final(normal_GB_biaoqiao); % 正常粒球中心 (cell)
normal_Radii = Radii_final(normal_GB_biaoqiao);     % 正常粒球半径

% 4.2.1 粒球中心距矩阵
GBs_distance = zeros(num_normal_balls, num_normal_balls);
for i = 1:num_normal_balls
    for j = 1:num_normal_balls
        if i ~= j
            GBs_distance(i,j) = sqrt(sum((normal_Centers{i} - normal_Centers{j}).^2));
        end
    end
end

% 4.2.2 粒球总尺度矩阵（半径和）
zong_chidu = zeros(num_normal_balls, num_normal_balls);
for i = 1:num_normal_balls
    for j = 1:num_normal_balls
        if i ~= j
            zong_chidu(i,j) = normal_Radii(i) + normal_Radii(j);
        end
    end
end

% 4.2.3 归一化中心距矩阵
zhongxin_ju = zeros(num_normal_balls, num_normal_balls);
for i = 1:num_normal_balls
    for j = 1:num_normal_balls
        if i ~= j && zong_chidu(i,j) ~= 0
            zhongxin_ju(i,j) = GBs_distance(i,j) / zong_chidu(i,j);
        end
    end
end

alpha = 1.2; % 连接阈值因子，可以调整（1.0表示必须重叠，>1.0允许一定间隙）

s1 = zeros(length(normal_GB_biaoqiao), length(normal_GB_biaoqiao));
for i = 1:length(normal_GB_biaoqiao)
    for j = 1:length(normal_GB_biaoqiao)
        if i ~= j
            % 修改相似度计算：允许zhongxin_ju < alpha时都有连接
            overlap_degree = max(0, (alpha - zhongxin_ju(i,j)));
            if overlap_degree > 0
                s1(i,j) = overlap_degree / alpha; % 归一化到[0,1]
            end
        end
    end
end

%% ===================== 7. 运行全局感知的HCS聚类 =====================
fprintf('\n=== 运行全局感知HCS聚类（稳健版） ===\n');
min_cluster_size = 2; % 最小簇规模（避免过小簇）

% --- robust mapping: center -> nearest mode (反向映射) ---
normal_Centers_mat = [];
for k = 1:length(normal_Centers)
    center_vec = normal_Centers{k};
    if size(center_vec,1) > 1
        center_vec = center_vec';
    end
    normal_Centers_mat = [normal_Centers_mat; center_vec];
end

% 如果 modes 的数量为0，报错并退出
if isempty(modes)
    error('modes 为空，无法继续映射。请先确认 meanShiftClusterModes_kdtree_annotated 的输出。');
end

% 反向映射：对每个 center 找到最近的 mode（index）
Dists_center_mode = pdist2(normal_Centers_mat, modes); % n_centers x num_modes
[~, nearest_mode_for_center] = min(Dists_center_mode, [], 2);

% 选择被视为"靠近某个 mode"的中心作为 core（可选加入距离阈值筛选）
core_ball_indices = find( nearest_mode_for_center >= 1 & nearest_mode_for_center <= size(modes,1) );
core_ball_indices = unique(core_ball_indices); % 去重、升序
fprintf('识别为核心节点的粒球（反向映射得到，共 %d 个）：\n', length(core_ball_indices));
fprintf('%s\n', num2str(core_ball_indices'));

% --- 调用 HCS 聚类（在 normal ball 空间上运行） ---
[cluster_labels_normal, clusters_normal] = HCS_Clustering(s1, min_cluster_size, core_ball_indices);

fprintf('HCS 初始聚类（normal balls）得到 %d 个簇（不含噪声）\n', length(clusters_normal));

% ===================== 7.1 分析HCS聚类结果 =====================
fprintf('\n=== 分析HCS聚类结果 ===\n');

% 显示cluster_labels_normal的分布
unique_labels = unique(cluster_labels_normal);
fprintf('cluster_labels_normal中的标签（包含噪声标签0）：%s\n', mat2str(unique_labels'));

% 统计每个标签的数量（包括噪声）
label_counts = zeros(length(unique_labels), 1);
for i = 1:length(unique_labels)
    label = unique_labels(i);
    count = sum(cluster_labels_normal == label);
    label_counts(i) = count;
    if label == 0
        fprintf('  噪声（标签0）：%d 个粒球\n', count);
    else
        fprintf('  簇 %d：%d 个粒球\n', label, count);
    end
end

% 显示clusters_normal的内容
fprintf('\nclusters_normal中包含 %d 个簇：\n', length(clusters_normal));
for i = 1:length(clusters_normal)
    fprintf('  簇 %d：%d 个粒球（索引：%s）\n', i, length(clusters_normal{i}), ...
        mat2str(clusters_normal{i}(1:min(5, length(clusters_normal{i})))'));
    if length(clusters_normal{i}) > 5
        fprintf('    ... 等 %d 个粒球\n', length(clusters_normal{i}) - 5);
    end
end

% ===================== 7.2 处理HCS聚类中的噪声粒球 =====================
fprintf('\n=== 处理HCS聚类中的噪声粒球 ===\n');

% 获取噪声粒球的索引（在normal_GB_biaoqiao中的索引）
noise_ball_indices = find(cluster_labels_normal == 0);
fprintf('HCS聚类识别出 %d 个噪声粒球（标签为0）\n', length(noise_ball_indices));

if ~isempty(noise_ball_indices)
    fprintf('噪声粒球索引（在normal_GB_biaoqiao中的位置）：%s\n', mat2str(noise_ball_indices'));
    
    % 获取所有非噪声粒球（簇标签>0的粒球）
    non_noise_indices = find(cluster_labels_normal > 0);
    fprintf('非噪声粒球数量：%d\n', length(non_noise_indices));
    
    if isempty(non_noise_indices)
        fprintf('警告：所有粒球都被标记为噪声！无法为噪声粒球分配标签。\n');
        % 将所有噪声粒球分配到一个新的簇中
        cluster_labels_normal(noise_ball_indices) = 1;
        fprintf('将所有噪声粒球分配到簇 1\n');
        
        % 更新clusters_normal
        clusters_normal = {noise_ball_indices};
    else
        % 为每个噪声粒球分配最近的非噪声粒球所在簇的标签
        fprintf('开始为噪声粒球分配标签（基于最近非噪声粒球）...\n');
        
        % 获取所有粒球中心坐标（正常粒球）
        centers_mat = zeros(num_normal_balls, D);
        for i = 1:num_normal_balls
            centers_mat(i, :) = normal_Centers{i};
        end
        
        % 为每个噪声粒球找到最近的非噪声粒球
        for i = 1:length(noise_ball_indices)
            noise_idx = noise_ball_indices(i);  % 在normal_GB_biaoqiao中的索引
            
            % 计算噪声粒球到所有非噪声粒球的距离
            noise_center = centers_mat(noise_idx, :);
            
            % 方法1：计算到所有非噪声粒球中心的距离
            distances = zeros(length(non_noise_indices), 1);
            for j = 1:length(non_noise_indices)
                non_noise_idx = non_noise_indices(j);
                non_noise_center = centers_mat(non_noise_idx, :);
                distances(j) = sqrt(sum((noise_center - non_noise_center).^2));
            end
            
            % 找到最近的粒球
            [~, min_idx] = min(distances);
            nearest_non_noise_idx = non_noise_indices(min_idx);
            
            % 获取最近粒球的簇标签
            nearest_label = cluster_labels_normal(nearest_non_noise_idx);
            
            % 为噪声粒球分配该标签
            cluster_labels_normal(noise_idx) = nearest_label;
            
            % 将该噪声粒球添加到对应的簇中
            for k = 1:length(clusters_normal)
                if clusters_normal{k}(1) == nearest_label || ...
                   (length(clusters_normal{k}) > 0 && any(clusters_normal{k} == nearest_non_noise_idx))
                    % 找到对应的簇，添加噪声粒球
                    clusters_normal{k} = sort([clusters_normal{k}, noise_idx]);
                    break;
                end
            end
            
            if mod(i, 10) == 0 || i == length(noise_ball_indices)
                fprintf('  已处理 %d/%d 个噪声粒球\n', i, length(noise_ball_indices));
            end
        end
        
        fprintf('噪声粒球标签分配完成！\n');
        
        % 验证所有噪声粒球是否都已分配标签
        remaining_noise = sum(cluster_labels_normal == 0);
        if remaining_noise > 0
            fprintf('警告：仍有 %d 个噪声粒球未被分配标签\n', remaining_noise);
        end
    end
else
    fprintf('没有噪声粒球，所有粒球都已分配到簇中。\n');
end


% ===================== 7.4 最终统计 =====================
fprintf('\n=== HCS聚类最终结果 ===\n');
final_labels = unique(cluster_labels_normal);
final_labels = final_labels(final_labels > 0); % 排除标签0（如果有）

if isempty(final_labels)
    fprintf('警告：最终没有有效的簇！\n');
    num_final_clusters = 0;
else
    num_final_clusters = length(final_labels);
    fprintf('最终得到 %d 个簇：\n', num_final_clusters);
    for i = 1:num_final_clusters
        label = final_labels(i);
        count = sum(cluster_labels_normal == label);
        fprintf('  簇 %d：%d 个粒球\n', label, count);
    end
end
%% ===================== 为异常粒球与未分配点分配簇标签 =====================
% 目标：
% 1) 将之前被视为“异常”的单点粒球（outlier_GB_biaoqiao）分配到最近的已分配簇；
% 2) 将每个粒球的簇标签拓展到其包含的原始数据点；
% 3) 对仍未分配标签的点，按 (dist(point, center) - radius) 找最近粒球并赋其簇标签。

fprintf('\n=== 为异常粒球与未分配数据点分配簇标签 ===\n');

% 1) 准备粒球中心和半径矩阵（保证行向量）
centers_all = zeros(numBalls, D);
for i = 1:numBalls
    c = Centers_final{i};
    if size(c,1) > 1, c = c'; end
    centers_all(i, :) = c;
end
radii_all = Radii_final(:); % 确保列向量尺寸 numBalls x 1

% 2) 为所有粒球创建标签容器（初始为 0 表示未分配）
ball_labels_all = zeros(numBalls,1);

% 已有 normal 粒球对应的标签 (cluster_labels_normal) 长度为 num_normal_balls，
% 它对应的是 normal_GB_biaoqiao 中列出的原始粒球索引的顺序。
for i = 1:num_normal_balls
    orig_ball_idx = normal_GB_biaoqiao(i);     % 原始粒球索引（相对于 finalBalls）
    ball_labels_all(orig_ball_idx) = cluster_labels_normal(i);
end

% 3) 为每个异常（单点）粒球分配最近已标记簇
if ~isempty(outlier_GB_biaoqiao)
    fprintf('为 %d 个异常粒球分配最近簇标签...\n', length(outlier_GB_biaoqiao));
    % 构建 normal centers 矩阵（用于距离计算）
    normal_positions = normal_GB_biaoqiao(:); % 原始粒球索引列表
    normal_centers_mat = centers_all(normal_positions, :);        % num_normal_balls x D
    normal_labels_vec = cluster_labels_normal(:);                 % num_normal_balls x 1

    % 如果存在标签为0的 normal 粒球（理论上不应，但做保险），准备非零索引
    nonzero_normal_idx = find(normal_labels_vec > 0);

    for k = 1:length(outlier_GB_biaoqiao)
        out_idx = outlier_GB_biaoqiao(k);        % 原始粒球索引
        out_center = centers_all(out_idx, :);

        % 计算到 normal 粒球中心的欧式距离
        dists = sqrt(sum((normal_centers_mat - out_center).^2, 2)); % num_normal_balls x 1
        [~, minpos] = min(dists);
        assign_label = normal_labels_vec(minpos);

        % 如果所选的 normal 粒球标签为 0（异常情况），则选择最近的非零标签粒球
        if assign_label == 0
            if ~isempty(nonzero_normal_idx)
                [~, relmin] = min(dists(nonzero_normal_idx));
                minpos2 = nonzero_normal_idx(relmin);
                assign_label = normal_labels_vec(minpos2);
            else
                % 万一所有 normal 也都是 0（不太可能），则赋 1
                assign_label = 1;
            end
        end

        ball_labels_all(out_idx) = assign_label;
    end
    fprintf('异常粒球标签分配完成。\n');
else
    fprintf('没有异常粒球需要分配。\n');
end

% 4) 检查是否仍有未分配的粒球（极端情况）
unassigned_balls = find(ball_labels_all == 0);
if ~isempty(unassigned_balls)
    fprintf('警告：仍有 %d 个粒球未分配（将分配到最近已标记粒球）。\n', length(unassigned_balls));
    % 寻找最近的已分配粒球并借用其标签
    assigned_idx = find(ball_labels_all > 0);
    for ii = 1:length(unassigned_balls)
        b = unassigned_balls(ii);
        dists = sqrt(sum((centers_all(assigned_idx,:) - centers_all(b,:)).^2, 2));
        [~, mpos] = min(dists);
        ball_labels_all(b) = ball_labels_all(assigned_idx(mpos));
    end
end

% 5) 将粒球标签拓展到原始数据点：点的标签 = 所在粒球的标签
point_labels = zeros(N,1);
for b = 1:numBalls
    idxs = finalBalls{b};
    if ~isempty(idxs)
        point_labels(idxs) = ball_labels_all(b);
    end
end

% 6) 对仍未分配标签的数据点（点标签仍为0），基于 dist_pt_to_ball = ||x - center|| - radius 进行分配
unassigned_points = find(point_labels == 0);
if ~isempty(unassigned_points)
    fprintf('尚有 %d 个数据点未分配簇，按 (点到中心距离 - 半径) 进行最近粒球分配...\n', length(unassigned_points));

    % 预先把 radii_all 和 centers_all 准备好（已准备）
    for pi = 1:length(unassigned_points)
        p = unassigned_points(pi);
        x = X(p, :); % 1 x D

        % 计算到所有粒球的 (距离 - 半径)
        diffs = centers_all - x; % numBalls x D
        center_dists = sqrt(sum(diffs.^2, 2)); % numBalls x 1
        pt2ball_d = center_dists - radii_all;  % 可能为负（点在粒球内部）

        % 找到最小的值（优先选择点包含的粒球，因为会是最小的负值）
        [~, minball] = min(pt2ball_d);

        % 如果该粒球尚未被分配标签（极端情况），选择最近已标记粒球
        if ball_labels_all(minball) == 0
            candidates = find(ball_labels_all > 0);
            if ~isempty(candidates)
                [~, relmin] = min(pt2ball_d(candidates));
                minball = candidates(relmin);
            else
                % 万一没有已分配粒球（非常极端），赋 1
                ball_labels_all(minball) = 1;
            end
        end

        point_labels(p) = ball_labels_all(minball);
    end
    fprintf('未分配数据点处理完成。\n');
else
    fprintf('所有数据点均已通过粒球分配到簇。\n');
end

% 7) 简单统计与检查
num_assigned_points = sum(point_labels > 0);
fprintf('最终：%d / %d 个数据点已分配簇标签。\n', num_assigned_points, N);

%% ===================== 9. 评估聚类结果 =====================
fprintf('\n=== 评估聚类结果 ===\n');

% 确保true_labels是列向量
true_labels = true_labels(:);

% 调用评估函数
[acc, nmi, ari, f1_macro] = evaluation(true_labels, point_labels);

fprintf('聚类性能指标：\n');
fprintf('  ACC (准确率): %.4f\n', acc);
fprintf('  NMI (归一化互信息): %.4f\n', nmi);
fprintf('  ARI (调整兰德指数): %.4f\n', ari);
fprintf('  F1-macro (宏平均F1分数): %.4f\n', f1_macro);