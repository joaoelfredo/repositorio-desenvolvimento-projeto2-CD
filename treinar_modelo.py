"""
treinar_modelo.py — Treina o Ridge (melhor modelo, RMSLE CV = 0.1105) com o mesmo
pré-processamento de pipeline.py e salva como modelo_baseline.joblib.

Uso:
    python treinar_modelo.py
    python treinar_modelo.py --treino data/treino.csv
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

RANDOM_STATE = 42

# Mesma lista de colunas assimétricas do pipeline.py (prever_precos)
SKEWED_COLS = [
    'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
    'ScreenPorch', 'total_sf', 'total_porch_sf',
]


def preprocessar_treino(df_raw: pd.DataFrame):
    """
    Aplica exatamente o mesmo pré-processamento de prever_precos() em pipeline.py,
    mas nos dados de treino (remove outliers e extrai a variável alvo).
    """
    df = df_raw.copy()

    # Remover outliers extremos (casas enormes com preço muito baixo)
    df = df[~((df['GrLivArea'] > 4000) & (df['SalePrice'] < 300_000))]
    df = df.reset_index(drop=True)

    # Extrair variável alvo em log1p antes de qualquer transformação
    y = np.log1p(df['SalePrice'])

    # Remover Id e SalePrice
    df = df.drop(columns=['Id', 'SalePrice'], errors='ignore')

    # Mesmas colunas de baixo valor preditivo que pipeline.py remove
    cols_to_drop = [
        'PoolArea', 'PoolQC', 'MiscFeature', 'MiscVal',
        'Street', 'Utilities', 'LowQualFinSF',
        '3SsnPorch', 'KitchenAbvGr', 'GarageYrBlt',
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Nulos com significado de negócio (NaN = ausência da feature)
    cols_none = [
        'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    ]
    for col in cols_none:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    cols_zero = [
        'GarageArea', 'GarageCars',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea',
    ]
    for col in cols_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Nulos genuinamente faltantes
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

    if 'MasVnrType' in df.columns:
        df['MasVnrType'] = df['MasVnrType'].fillna('None')
    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna('SBrkr')
    if 'Functional' in df.columns:
        df['Functional'] = df['Functional'].fillna('Typ')

    # Tipos — categorias armazenadas como inteiros
    for col in ['MSSubClass', 'MoSold', 'YrSold']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Feature engineering
    yr = df['YrSold'].astype(int)
    df['house_age']         = yr - df['YearBuilt']
    df['years_since_remod'] = yr - df['YearRemodAdd']
    df['was_remodeled']     = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
    df['total_sf']          = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['total_bathrooms']   = (df['FullBath'] + df['BsmtFullBath']
                               + 0.5 * df['HalfBath'] + 0.5 * df['BsmtHalfBath'])
    df['total_porch_sf']    = df['OpenPorchSF'] + df['EnclosedPorch'] + df['ScreenPorch']
    df['has_fireplace']     = (df['Fireplaces'] > 0).astype(int)
    df['has_garage']        = (df['GarageArea'] > 0).astype(int)
    df['has_pool']          = 0  # PoolArea já foi removido

    # Encoding ordinal
    qual_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    for col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']:
        if col in df.columns:
            df[col] = df[col].map(qual_map).fillna(0)

    if 'BsmtExposure' in df.columns:
        df['BsmtExposure'] = df['BsmtExposure'].map(
            {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
        ).fillna(0)

    bsmt_fin_map = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
    for col in ['BsmtFinType1', 'BsmtFinType2']:
        if col in df.columns:
            df[col] = df[col].map(bsmt_fin_map).fillna(0)

    if 'GarageFinish' in df.columns:
        df['GarageFinish'] = df['GarageFinish'].map(
            {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
        ).fillna(0)

    if 'Functional' in df.columns:
        df['Functional'] = df['Functional'].map(
            {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7}
        ).fillna(7)

    if 'PavedDrive' in df.columns:
        df['PavedDrive'] = df['PavedDrive'].map({'N': 0, 'P': 1, 'Y': 2}).fillna(0)

    if 'LandSlope' in df.columns:
        df['LandSlope'] = df['LandSlope'].map({'Gtl': 0, 'Mod': 1, 'Sev': 2}).fillna(0)

    if 'CentralAir' in df.columns:
        df['CentralAir'] = df['CentralAir'].map({'N': 0, 'Y': 1}).fillna(0)

    # One-Hot Encoding (mesmas colunas que pipeline.py)
    onehot_cols = [
        'MSSubClass', 'MSZoning', 'Alley', 'LotShape', 'LandContour',
        'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
        'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
        'Heating', 'Electrical', 'GarageType', 'Fence',
        'SaleType', 'SaleCondition', 'MoSold', 'YrSold',
    ]
    onehot_cols = [c for c in onehot_cols if c in df.columns]
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=False, dtype=int)

    # log1p nas features assimétricas (mesma lista de pipeline.py)
    for col in SKEWED_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    return df, y


def treinar(caminho_treino: str = 'data/treino.csv') -> tuple:
    print(f'Carregando: {caminho_treino}')
    df_raw = pd.read_csv(caminho_treino)

    print('Pre-processando (identico ao pipeline.py)...')
    X, y = preprocessar_treino(df_raw)
    print(f'Shape: X={X.shape}, y={y.shape}')

    modelo = Ridge(alpha=10)

    print('Validacao cruzada (5-fold, RMSLE)...')
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = -cross_val_score(
        modelo, X, y,
        scoring='neg_root_mean_squared_error',
        cv=kf, n_jobs=-1,
    )
    print(f'RMSLE CV medio: {scores.mean():.5f}  +/-{scores.std():.5f}')
    print(f'Scores por fold: {[round(s, 5) for s in scores]}')

    print('Treinando modelo final com todos os dados...')
    modelo.fit(X, y)

    joblib.dump(modelo, 'modelo_baseline.joblib')
    print(f'Modelo salvo: modelo_baseline.joblib')

    return modelo, float(scores.mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treina o modelo e salva modelo_baseline.joblib')
    parser.add_argument('--treino', default='data/treino.csv',
                        help='Caminho para o CSV de treino')
    args = parser.parse_args()

    modelo, rmsle = treinar(args.treino)
    print(f'\nPronto! RMSLE estimado (CV 5-fold): {rmsle:.5f}')
    print('Execute python pipeline.py para validar o pipeline localmente.')
