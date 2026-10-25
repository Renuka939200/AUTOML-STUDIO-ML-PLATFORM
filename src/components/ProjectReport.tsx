import React from 'react';
import {
  BookOpen,
  FileText,
  Cpu,
  Layers,
  CheckCircle2,
  AlertTriangle,
  Award,
  Zap,
  Activity,
  ShieldCheck,
  Code2
} from 'lucide-react';

export const ProjectReport: React.FC = () => {
  return (
    <div className="max-w-5xl mx-auto space-y-8 pb-12 text-slate-300">
      {/* Hero Title */}
      <div className="p-8 rounded-3xl bg-gradient-to-br from-slate-900 via-indigo-950/40 to-slate-900 border border-slate-800 shadow-lg">
        <div className="flex items-center space-x-2 text-indigo-400 text-xs font-bold uppercase tracking-wider mb-2">
          <BookOpen className="w-4 h-4" />
          <span>Technical Report & Architectural Documentation</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-black text-white tracking-tight">
          AutoML Studio: End-to-End Automated Machine Learning Platform
        </h1>
        <p className="text-sm text-slate-400 mt-2 leading-relaxed max-w-3xl">
          An end-to-end full-stack machine learning workbench integrating automated CSV preprocessing, Scikit-learn pipelines, cross-validation hyperparameter fine-tuning, statistical evaluation, Joblib model persistence, and real-time inference.
        </p>

        <div className="mt-6 pt-6 border-t border-slate-800/80 grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
          <div>
            <span className="text-slate-500 block">Core ML Framework:</span>
            <strong className="text-white">Scikit-learn (Python)</strong>
          </div>
          <div>
            <span className="text-slate-500 block">Data Wrangling:</span>
            <strong className="text-white">Pandas & NumPy</strong>
          </div>
          <div>
            <span className="text-slate-500 block">Model Persistence:</span>
            <strong className="text-white">Joblib (.joblib)</strong>
          </div>
          <div>
            <span className="text-slate-500 block">Metadata Registry:</span>
            <strong className="text-white">SQLite (automl.db)</strong>
          </div>
        </div>
      </div>

      {/* 1. Problem Definition & Architecture */}
      <section className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Layers className="w-5 h-5 text-blue-400" />
          1. Problem Definition & Architectural Flow
        </h2>
        <p className="text-xs leading-relaxed text-slate-400">
          Machine learning workflows often suffer from boilerplate overhead, unintentional data leakage (such as fitting scalers before train-test splitting), improper task selection, and complex model deployment. AutoML Studio solves these problems by providing an automated web interface while maintaining rigorous mathematical standards.
        </p>

        <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 font-mono text-xs text-blue-300">
          User Upload (CSV) → Schema Inference & Deduplication → Target & Feature Selection → Leak-Free Preprocessor (SimpleImputer + StandardScaler + OneHotEncoder) → Baseline Estimator Fit → K-Fold Hyperparameter Tuning (GridSearchCV / RandomizedSearchCV) → Test Split Scoring (Confusion Matrix, ROC-AUC, R², RMSE) → Joblib Serialization → Interactive & Batch Inference
        </div>
      </section>

      {/* 2. Data Preprocessing & Leakage Prevention */}
      <section className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-emerald-400" />
          2. Data Preprocessing & Leakage Prevention
        </h2>
        <p className="text-xs leading-relaxed text-slate-400">
          A critical flaw in naive AutoML implementations is <em>lookahead data leakage</em>, where transformations compute statistics across the entire dataset prior to splitting. AutoML Studio guarantees mathematical integrity:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1.5">
            <span className="font-bold text-white block">Automated Imputation</span>
            <p className="text-slate-400">
              Numerical columns use median or mean imputation; categorical columns use most-frequent imputation. Imputation statistics are learned strictly from the training partition.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1.5">
            <span className="font-bold text-white block">One-Hot Encoding</span>
            <p className="text-slate-400">
              Categorical features are encoded via <code className="text-indigo-300 font-mono">OneHotEncoder(handle_unknown='ignore')</code>, ensuring unseen inference categories do not crash the pipeline.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1.5">
            <span className="font-bold text-white block">Feature Standardization</span>
            <p className="text-slate-400">
              Zero-mean, unit-variance scaling (<code className="text-indigo-300 font-mono">StandardScaler</code>) preserves numeric scale balance for distance-sensitive algorithms like SVM and KNN.
            </p>
          </div>
        </div>
      </section>

      {/* 3. Machine Learning Algorithms Implemented */}
      <section className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Cpu className="w-5 h-5 text-indigo-400" />
          3. Implemented Machine Learning Algorithms
        </h2>
        <p className="text-xs leading-relaxed text-slate-400">
          The platform includes 11 distinct Scikit-learn algorithms spanning linear, ensemble, kernel, and instance-based paradigms:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          {/* Classification Suite */}
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 space-y-2">
            <span className="font-bold text-purple-300 uppercase tracking-wider block">
              Classification Algorithms (6 Models)
            </span>
            <ul className="space-y-1.5 text-slate-400 list-disc list-inside">
              <li><strong className="text-white">Logistic Regression:</strong> L-BFGS optimizer with L2 penalty.</li>
              <li><strong className="text-white">Decision Tree Classifier:</strong> Recursive binary splitting via Gini / Entropy.</li>
              <li><strong className="text-white">Random Forest Classifier:</strong> Bagging ensemble of B decision trees with random subspace sampling.</li>
              <li><strong className="text-white">K-Nearest Neighbors (KNN):</strong> Non-parametric Euclidean neighborhood voting.</li>
              <li><strong className="text-white">Naive Bayes:</strong> GaussianNB assuming class-conditional feature independence.</li>
              <li><strong className="text-white">Support Vector Machine (SVC):</strong> Maximal margin hyperplane with RBF kernel and Platt probability scaling.</li>
            </ul>
          </div>

          {/* Regression Suite */}
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 space-y-2">
            <span className="font-bold text-emerald-300 uppercase tracking-wider block">
              Regression Algorithms (5 Models)
            </span>
            <ul className="space-y-1.5 text-slate-400 list-disc list-inside">
              <li><strong className="text-white">Linear Regression:</strong> Ordinary Least Squares (OLS) minimizing sum of squared residuals.</li>
              <li><strong className="text-white">Polynomial Regression:</strong> PolynomialFeatures degree expansion combined with linear modeling.</li>
              <li><strong className="text-white">Decision Tree Regressor:</strong> Minimizes mean squared error across terminal leaf segments.</li>
              <li><strong className="text-white">Random Forest Regressor:</strong> Ensembled variance reduction over decorrelated regression trees.</li>
              <li><strong className="text-white">Support Vector Regression (SVR):</strong> Epsilon-insensitive tube regression with RBF kernel.</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 4. Hyperparameter Fine-Tuning */}
      <section className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Zap className="w-5 h-5 text-amber-400" />
          4. Hyperparameter Tuning Methodology
        </h2>
        <p className="text-xs leading-relaxed text-slate-400">
          The tuning module utilizes <code className="text-amber-300 font-mono">GridSearchCV</code> or <code className="text-amber-300 font-mono">RandomizedSearchCV</code> across K-fold cross-validation folds. Crucially, the search is conducted strictly within the training set, preserving the test split as an unbiased estimate of generalization error. The best parameter combination is selected based on mean cross-validation score.
        </p>
      </section>

      {/* 5. Limitations & Future Roadmap */}
      <section className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-rose-400" />
          5. Known Limitations & Future Work
        </h2>
        <div className="space-y-2 text-xs text-slate-400">
          <p>
            • <strong>High-Cardinality Categoricals:</strong> Categorical variables with hundreds of unique text values can expand feature dimensionality significantly when one-hot encoded. Future versions can incorporate target encoding or entity embeddings.
          </p>
          <p>
            • <strong>Imbalanced Datasets:</strong> Currently handled via stratified splitting; future iterations could support SMOTE oversampling or focal loss weighting.
          </p>
          <p>
            • <strong>Automated Ensemble Stacking:</strong> Support for StackingClassifier / VotingRegressor across the top 3 best performing estimators.
          </p>
        </div>
      </section>
    </div>
  );
};
