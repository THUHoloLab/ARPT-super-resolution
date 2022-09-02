function [SR, model, report] = reweightedOptimizationSR(LRImages, model, optimParams, varargin)
%% the 
tic
    if nargin < 3
        % Get default optimization parameters.
        optimParams = getReweightedOptimizationParams;
    end
    if nargin > 3
        % Get default optimization parameters.
        if varargin{1}==1
        useFixedRegularizationWeight=1;
        elseif varargin{1}==0
        useFixedRegularizationWeight=0;
        end
    end
    
    if nargin > 4
        % Use ground truth image provided by the user for synthetic data
        % experiments.
        groundTruth = varargin{2};
    else
        % No ground truth available.
        groundTruth = [];
    end
    
    if nargout > 2
%         Setup the report structure.
        report.SR = {[]};
        report.sigmaNoise = [];
        report.sigmaPrior = [];
        report.regularizationWeight = [];
        report.valError = {[]};
        report.trainError = {[]};
        report.numFunEvals = 0;
        report.observationWeights = {};
        report.priorWeights = {};
        report.data_term = [];
        report.prior_term = [];
        report.time=[];
    end
%     useFixedRegularizationWeight=1;
    %********** Initialization
    % Setup the coarse-to-fine optimization parameters.
    if ~isempty(optimParams.numCoarseToFineLevels)
        % Get user-defined number of coarse-to-fine levels to build the
        % image pyramid.
        numCoarseToFineLevels = optimParams.numCoarseToFineLevels;
        coarseToFineScaleFactors = max([1 (model.magFactor - numCoarseToFineLevels + 1)]) : model.magFactor;
    else
        % Use default settings for coarse-to-fine optimization. Build image
        % pyramid with integer scales between 1 and the desired
        % magnification factor.
        coarseToFineScaleFactors = min([1 model.magFactor]) : model.magFactor;
    end
    if isempty(model.SR)
        % Initialize super-resolved image by the temporal median of the 
        % motion-compensated low-resolution frames.
        SR = imageToVector( imresize(medfilttemp(LRImages, model.motionParams), coarseToFineScaleFactors(1)) );
    else
        % Use the user-defined initial guess. This image needs to be
        % resized to the coarsest level of the image pyramid.
        SR = imageToVector( imresize(model.SR, coarseToFineScaleFactors(1) * size(LRImages(:,:,1))) );
    end   
    % Initialize the confidence weights of the observation model.
    for frameIdx = 1:size(LRImages, 3)
        % Use uniform weights as initial guess.
        observationWeights{frameIdx} = ones(numel(LRImages(:,:,frameIdx)), 1);
    end
    if isempty(model.confidence)
        model.confidence = observationWeights;
    end
    observationWeightsStatic = model.confidence;
    
    
    
    
    %%
    % Iterations for cross validation based hyperparameter selection.
    maxCVIter = optimParams.maxCVIter;
 
    objFun=[];coe_prior=6;%5.5;
    % Main optimization loop of iteratively re-weighted minimization.
    for iter = 1 : optimParams.maxMMIter%10%
            
        %********** Extract current level from image pyramid.
        if iter <= length(coarseToFineScaleFactors)            
            % Assemble the system matrices for this level of the image
            % pyramid.
            model.magFactor = coarseToFineScaleFactors(iter);
            for frameIdx = 1:size(LRImages, 3)
                W{frameIdx} = composeSystemMatrix(size(LRImages(:,:,frameIdx)), model.magFactor, model.psfWidth, model.motionParams{frameIdx});
                Wt{frameIdx} = W{frameIdx}';
            end
            
            % Propagate the estimate to this level of the image pyramid.
            if iter > 1
                SR = imresize(vectorToImage(SR, coarseToFineScaleFactors(iter-1) * size(LRImages(:,:,1))), coarseToFineScaleFactors(iter)*size(LRImages(:,:,1)));
                SR = imageToVector(SR);
            end      
        end
        % else: we have already reached the desired magnification level.
%%
        %********** Update the observation confidence weights.
        sigmaNoise = 0;
        if ~isempty(optimParams.observationWeightingFunction)
            for frameIdx = 1:size(LRImages, 3)
                % Compute the residual error.
                y{frameIdx} = imageToVector(LRImages(:,:,frameIdx));
                if ~isempty(model.photometricParams)
                    if ~isvector(model.photometricParams.mult)
                        residualError{frameIdx} = getResidualForSingleFrame(SR, y{frameIdx}, W{frameIdx}, model.photometricParams.mult(:,:,frameIdx), model.photometricParams.add(:,:,frameIdx));
                    else
                        residualError{frameIdx} = getResidualForSingleFrame(SR, y{frameIdx}, W{frameIdx}, model.photometricParams.mult(frameIdx), model.photometricParams.add(frameIdx));
                    end
                else
                    residualError{frameIdx} = getResidualForSingleFrame(SR, y{frameIdx}, W{frameIdx});
                end
            end
            [observationWeights, sigmaNoise] = optimParams.observationWeightingFunction.function(residualError, model.confidence, optimParams.observationWeightingFunction.parameters{1:end});

            % Combine the dynamic observation weights with static,
            % user-defined confidence weights.
            for frameIdx = 1:size(LRImages, 3)
%                 model.confidence{frameIdx} =ones(size( observationWeightsStatic{frameIdx} .* observationWeights{frameIdx}));
                model.confidence{frameIdx} = observationWeightsStatic{frameIdx} .* observationWeights{frameIdx};
            end
            
        % else: use uniform weights if no weighting function provided
        end
        %********** Update the image prior confidence weights.
        sigmaPrior = 0;
        if ~isempty(optimParams.priorWeightingFunction)
            % Apply sparsity transform to the current estimate of the
            % high-resolution image according to the image prior.
            model.imagePrior.parameters{1} = model.magFactor * size(LRImages(:,:,1));
            [~, transformedImage] =  model.imagePrior.function(SR, model.imagePrior.parameters{1:end-1});
            % Filtering of the sparsity transformed image.
            transformedImageFiltered = [];
            if iscell(transformedImage)
                for l = 1:size(transformedImage,1)
                    for m = 1:size(transformedImage,2)
                        if ~isempty(transformedImage{l,m})
                            z = vectorToImage( transformedImage{l,m}, model.magFactor * size(LRImages(:,:,1)));
                            transformedImageFiltered = [transformedImageFiltered; imageToVector( medfilt2(z, [3 3]) )];
                        end
                    end
                end
            else
                transformedImageFiltered = imageToVector( medfilt2(transformedImage, [3 3]) );
            end
            if ~exist('priorWeights', 'var')
                % Initialize weights at the first iteration.
                priorWeights = ones(size(transformedImageFiltered));
            end
            if numel(priorWeights) ~= numel(transformedImageFiltered)
                % Propagate the weights from the previous level in
                % coarse-to-fine optimization to the current level.
                priorWeights = imresize(priorWeights, size(transformedImageFiltered));
            end
            [priorWeights, sigmaPrior] = optimParams.priorWeightingFunction.function(transformedImageFiltered, priorWeights, optimParams.priorWeightingFunction.parameters{1:end});
        else
            % Use uniform weights if no weighting function provided.
            priorWeights = 1;
        end
        model.imagePrior.parameters{end} = priorWeights;

        %% ********** Hyperparameter selection
        if maxCVIter > 1 && ~useFixedRegularizationWeight
            % Automatic hyperparameter selection using cross validation.
            [model.imagePrior.weight, SR_best] = selectRegularizationWeight;
            % Update number of cross validation iterators.
            maxCVIter = max([round( 0.5 * maxCVIter ) 1]); 
            [data_term,prior_term] = calculate_data_prior_term(SR, model, y, W);
            VarationCoePrior=sigmaPrior/prior_term;
        else 
            % adaptive regularization parameter tuning （ARPT） algorithm.
            SR_best = SR;
            [data_term,prior_term] = calculate_data_prior_term(SR, model, y, W);
            if iter==1
                [data_term,prior_term] = calculate_data_prior_term(SR, model, y, W);
                sigmaNoise_original=sigmaNoise;
            end
            VarationCoePrior=sigmaPrior/prior_term;
            if iter>model.magFactor && objFun(iter)>objFun(iter-1)%data_term>data_term_old %(|| model.imagePrior.weight>report.regularizationWeight(iter-2))%&&*prior_term >*prior_term_old%objFun(iter)>objFun(iter-1)
                 coe_prior=0.9*coe_prior;
            end
            model.imagePrior.weight  = log(1+14*data_term).*sigmaNoise + coe_prior*(1*exp(- 80*VarationCoePrior) + 25*exp(-50*prior_term) )*sigmaNoise*prior_term;
        end

    

        if iter==1
        objFun(iter)=(data_term+model.imagePrior.weight*prior_term)*numel(SR);
        end
        %********** Update estimate for the high-resolution image.
        SR_old = SR;%SR_best
        [SR, numFunEvals,objFun(iter+1)] = updateHighResolutionImage(SR_old, model, y, W, Wt);%,data_term,prior_term
        data_term_old=data_term;prior_term_old=prior_term;
        %********** Check for convergence.
        if isConverged(SR, SR_old) && (iter > length(coarseToFineScaleFactors))
            % Convergence tolerance reached.
            SR = vectorToImage(SR, model.magFactor * size(LRImages(:,:,1)));
            return;
        end
        
        if nargout > 2
            % Log results for current iteration.
            report.SR{iter} = vectorToImage(SR, model.magFactor * size(LRImages(:,:,1)));
            report.numFunEvals = report.numFunEvals + numFunEvals;
            report.sigmaNoise(iter) = sigmaNoise;
            report.sigmaPrior(iter) = sigmaPrior;
            report.regularizationWeight(iter) = model.imagePrior.weight;
            report.observationWeights = cat(1, report.observationWeights, model.confidence);
            report.priorWeights = cat(1, report.priorWeights, priorWeights);
            report.data_term(iter)=data_term;
            report.prior_term(iter)=prior_term; 
            report.VarationCoePrior(iter)=VarationCoePrior;
            report.objFun(iter)=objFun(iter);
            if ~isempty(groundTruth)
                % Measure PSNR and SSIM for the given ground truth image.
                
                eccParams.iterations = 30;
                eccParams.levels = 2;
                eccParams.transform =  'translation';
                temp=imresize(vectorToImage(SR, model.magFactor * size(LRImages(:,:,1))), size(groundTruth));
                H = iat_ecc(temp, groundTruth, eccParams);
                temp=imtranslate(temp,-[H(1),H(2)]);
                report.psnr(iter) = psnr(temp(11:end-10,11:end-10), groundTruth(11:end-10,11:end-10));
                report.ssim(iter) = ssim(temp(11:end-10,11:end-10), groundTruth(11:end-10,11:end-10));
%                 report.psnr(iter) = psnr(imresize(vectorToImage(SR, model.magFactor * size(LRImages(:,:,1))), size(groundTruth)), groundTruth);
%                 report.ssim(iter) = ssim(imresize(vectorToImage(SR, model.magFactor * size(LRImages(:,:,1))), size(groundTruth)), groundTruth);
                report.time(iter) = toc;
            end
        end
        
    end
    
    SR = vectorToImage(SR, model.magFactor * size(LRImages(:,:,1)));
    
    function [SR, numIters,objFun,data_term,prior_term] = updateHighResolutionImage(SR, model, y, W, Wt)
        
        % Setup parameters for SCG optimization.
        scgOptions = zeros(1,18);
        scgOptions(2) = optimParams.terminationTol;
        scgOptions(3) = optimParams.terminationTol;
        scgOptions(10) = optimParams.maxSCGIter;
        scgOptions(14) = optimParams.maxSCGIter;
        
        % Perform SCG iterations to update the current estimate of the
        % high-resolution image.
        if iscolumn(SR)
            SR = SR';
        end
        [SR, ~, flog,~,~,data_term, prior_term] = scg(@imageObjectiveFunc, SR, scgOptions, @imageObjectiveFunc_grad, model, y, W, Wt);
        numIters = length(flog);
        SR = SR';
        objFun=flog(end);
    end

    function [bestLambda, SR_best] = selectRegularizationWeight
        
        % Split the set of given observations into training and validation
        % subset.
        for k = 1:size(LRImages, 3)
            fractionCvTrainingObservations = optimParams.fractionCVTrainingObservations;
            trainObservations{k} = 1 - (randn(size(y{k})) > fractionCvTrainingObservations);  %#ok<*AGROW>
            y_train{k}  = y{k}(trainObservations{k} == 1);
            y_val{k}    = y{k}(trainObservations{k} == 0);
            W_train{k}  = W{k}(trainObservations{k} == 1,:);
            Wt_train{k} = W_train{k}';
            W_val{k}    = W{k}(trainObservations{k} == 0,:);
            if ~isempty(model.photometricParams)
                if isvector(model.photometricParams.mult)
                    gamma_m_train{k}    = model.photometricParams.mult(k);
                    gamma_m_val{k}      = model.photometricParams.mult(k);
                    gamma_a_train{k}    = model.photometricParams.add(k);
                    gamma_a_val{k}      = model.photometricParams.add(k);
                else
                    gamma_m = model.photometricParams.mult(:,:,k);
                    gamma_m_train{k}    = gamma_m(trainObservations{k} == 1);
                    gamma_m_val{k}      = gamma_m(trainObservations{k} == 0);
                    gamma_a             = model.photometricParams.add(:,:,k);
                    gamma_a_train{k}    = gamma_a(trainObservations{k} == 1);
                    gamma_a_val{k}      = gamma_a(trainObservations{k} == 0);
                end
            else
                gamma_m_train{k}    = [];
                gamma_m_val{k}      = [];
                gamma_a_train{k}    = [];
                gamma_a_val{k}      = [];
            end
        end
        
        % Setup the model structure for the training subset.
        parameterTrainingModel = model;
        for k = 1:size(LRImages, 3)
            observationConfidenceWeights = model.confidence{k};
            parameterTrainingModel.confidence{k} = observationConfidenceWeights(trainObservations{k} == 1);
        end
        
        % Define search range for adaptive grid search.
        if ~isempty(model.imagePrior.weight)
            % Refine the search range from the previous iteration.
            lambdaSearchRange = logspace(log10(model.imagePrior.weight) - 1/iter, log10(model.imagePrior.weight) + 1/iter, maxCVIter);
        else
            % Set search range used for initialization.
            lambdaSearchRange = logspace(optimParams.hyperparameterCVSearchRange(1), optimParams.hyperparameterCVSearchRange(2), maxCVIter);
            bestLambda = median(lambdaSearchRange);
        end
        
        % Perform adaptive grid search over the selected search range.
        SR_best = SR;
        minValError = Inf;
        if exist('report', 'var')
            report.valError{iter} = [];
            report.trainError{iter} = [];
        end
        for lambda = lambdaSearchRange
                       
            % Estimate super-resolved image from the training set.
            parameterTrainingModel.imagePrior.weight = lambda;
            [SR_train, numFunEvals] = updateHighResolutionImage(SR, parameterTrainingModel, y_train, W_train, Wt_train);
            
            % Determine errors on the training and the validation subset.
            valError = 0;
            trainError = 0;
            for k = 1:size(LRImages, 3)
                observationConfidenceWeights = model.confidence{k};
                % Error on the validation subset.
                rk_val = getResidualForSingleFrame(SR_train, y_val{k}, W_val{k}, gamma_m_val{k}, gamma_a_val{k});
                valError = valError + sum( observationConfidenceWeights(trainObservations{k} == 0) .* (rk_val.^2) );
                % Error on the training subset.
                rk_train = getResidualForSingleFrame(SR_train, y_train{k}, W_train{k}, gamma_m_train{k}, gamma_a_train{k});
                trainError = trainError + sum( observationConfidenceWeights(trainObservations{k} == 1) .* (rk_train.^2) );
            end
            if valError < minValError
                % Found optimal regularization weight.
                bestLambda = lambda;
                minValError = valError;
                SR_best = SR_train;
            end
            
            if exist('report', 'var')
                report.numFunEvals = report.numFunEvals + length(numFunEvals);
                % Save errors on training and validation sets.
                report.valError{iter} = cat(1, report.valError{iter}, valError);
                report.trainError{iter} = cat(1, report.trainError{iter}, trainError);
            end
        end
        
    end

    function converged = isConverged(SR, SR_old)        
        converged = (max(abs(SR_old - SR)) <optimParams.terminationTol); %1e-4
    end
    
end
    
function [f,dataTerm, priorTerm]= imageObjectiveFunc(SR, model, y, W, ~)
    if ~iscolumn(SR)
        % Reshape to column vector. 
        SR = SR';
    end
    
    % Evaluate the data fidelity term.
    dataTerm = 0;
    for k = 1:length(y)
        if ~isempty(model.photometricParams)
            if isvector(model.photometricParams.mult)
                rk = getResidualForSingleFrame(SR, y{k}, W{k}, model.photometricParams.mult(k), model.photometricParams.add(k));
            else
                rk = getResidualForSingleFrame(SR, y{k}, W{k}, model.photometricParams.mult(:,:,k), model.photometricParams.add(:,:,k));
            end
        else
            rk = getResidualForSingleFrame(SR, y{k}, W{k});
        end
        dataTerm = dataTerm + sum( model.confidence{k} .* (rk.^2) );
    end
    % Evaluate image prior for regularization the super-resolved estimate.
    priorTerm = model.imagePrior.function(SR, model.imagePrior.parameters{1:end});
    % Calculate objective function.
    f = dataTerm + model.imagePrior.weight * priorTerm;
    dataTerm=dataTerm/numel(SR);
    priorTerm=priorTerm/numel(SR);
end
                
function [grad,dataTerm_grad,priorTerm_grad] = imageObjectiveFunc_grad(SR, model, y, W, Wt)
    
    if ~iscolumn(SR)
        % Reshape to column vector. 
        SR = SR';
    end
    
    % Calculate gradient of the data fidelity term w.r.t. the
    % super-resolved image.
    dataTerm_grad = 0;
    for k = 1:length(y)
        if ~isempty(model.photometricParams)
            if isvector(model.photometricParams.mult)
                rk = getResidualForSingleFrame(SR, y{k}, W{k}, model.photometricParams.mult(k), model.photometricParams.add(k));
            else
                rk = getResidualForSingleFrame(SR, y{k}, W{k}, model.photometricParams.mult(:,:,k), model.photometricParams.add(:,:,k));
            end
        else
            rk = getResidualForSingleFrame(SR, y{k}, W{k});
        end
        dataTerm_grad = dataTerm_grad - 2*Wt{k} * ( model.confidence{k} .* rk ); 
    end
    
    % Calculate gradient of the regularization term w.r.t. the 
    % super-resolved image.
    priorTerm_grad = model.imagePrior.gradient(SR, model.imagePrior.parameters{1:end});
    
    % Sum up to total gradient
    grad = dataTerm_grad + model.imagePrior.weight * priorTerm_grad;
    grad = grad';
    
end

function r = getResidualForSingleFrame(x, y, W, gamma_m, gamma_a)
    if nargin < 4 || isempty(gamma_m)
        gamma_m = 1;
    end
    if nargin < 5 || isempty(gamma_a)
        gamma_a = 0;
    end
    r = (y - (gamma_m .* (W*x) + gamma_a));
end
    
function [data_term,prior_term] = calculate_data_prior_term(SR,model, y, W, ~)
    if ~iscolumn(SR)
        % Reshape to column vector. 
        SR = SR';
    end
    % Evaluate the data fidelity term.
    dataTerm = 0;
    for k = 1:length(y)
        if ~isempty(model.photometricParams)
            if isvector(model.photometricParams.mult)
                rk = getResidualForSingleFrame(SR, y{k}, W{k}, model.photometricParams.mult(k), model.photometricParams.add(k));
            else
                rk = getResidualForSingleFrame(SR, y{k}, W{k}, model.photometricParams.mult(:,:,k), model.photometricParams.add(:,:,k));
            end
        else
            rk = getResidualForSingleFrame(SR, y{k}, W{k});
        end
%         dataTerm = dataTerm + sum( model.confidence{k} .* (abs(rk)) );
        dataTerm = dataTerm + sum( model.confidence{k} .* (rk.^2) );
    end
    % Evaluate image prior for regularization the super-resolved estimate.
        priorTerm = medfilt_btvPriorWeighted(SR, model.imagePrior.parameters{1:end});
        data_term=dataTerm/numel(SR);
        prior_term=priorTerm/numel(SR);
    
end  

function [dataTerm_grad,priorTerm_grad] = imageObjectiveFunc_grad1(SR, model, y, W, Wt)
    
    if ~iscolumn(SR)
        Reshape to column vector. 
        SR = SR';
    end
    
    % Calculate gradient of the data fidelity term w.r.t. the
    % super-resolved image.
    dataTerm_grad = 0;
    for k = 1:length(y)
        if ~isempty(model.photometricParams)
            if isvector(model.photometricParams.mult)
                rk = getResidualForSingleFrame(SR, y{k}, W{k}, model.photometricParams.mult(k), model.photometricParams.add(k));
            else
                rk = getResidualForSingleFrame(SR, y{k}, W{k}, model.photometricParams.mult(:,:,k), model.photometricParams.add(:,:,k));
            end
        else
            rk = getResidualForSingleFrame(SR, y{k}, W{k});
        end
        dataTerm_grad = dataTerm_grad - 2*Wt{k} * ( model.confidence{k} .* rk ); 
    end
    % Calculate gradient of the regularization term w.r.t. the 
    % super-resolved image.
    priorTerm_grad = model.imagePrior.gradient(SR, model.imagePrior.parameters{1:end});
end
function scaleParameter = getAdaptiveScaleParameter(z, weights)
    scaleParameter = weightedMedian( abs(z - weightedMedian(z, weights)), weights );
end