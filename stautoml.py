# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 20:53:10 2021

PycaretとStreamlitで作るGUI AutoML
"""

# streamlit run filename

import streamlit as st
import pandas as pd
import datetime

st.markdown("# 1. データをアップロードします")
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type='csv', key='train')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown("# 2. アップロードされたデータを確認します")
    st.dataframe(df.head(10))
    
    st.markdown("# 3. ターゲットを入力してください")
    target = st.text_input(label='ターゲット名を文字列で正しく入力してください', value=df.columns[-1])
    
    st.markdown("# 4. 回帰の場合はregression、分類の場合はclassificationを選択してください")
    ml_usecase = st.selectbox(label='regressionかclassificationを選択してください',
                              options=('', 'regression', 'classification'),
                              key='ml_usecase')
    if ml_usecase == 'regression':
        from pycaret.regression import *
    elif ml_usecase == 'classification':
        from pycaret.classification import *
    else:
        st.text('「regressionか「classification」か選択してください')  
    
    if (ml_usecase == 'regression') | (ml_usecase == 'classification'):
    
        st.markdown("# 5. 実行します")
    #compare_button = st.button(label='実行', key='compare_model')
    #if compare_button:
        st.markdown("実行中です…しばらくお待ち下さい")
        ml = setup(data=df,
                   target=target,
                   session_id=1234,
                   silent=True,
        )
        
        best = compare_models()  # 
        st.dataframe(best)
        
        st.markdown("# 6. モデルを選択してください。")
        select_model = st.selectbox(label='モデルを選択してください',
                              options=tuple(best.index),
                              key='select_model')
        save_button = st.button(label='モデル構築', key='save_model')
        if save_button:
            model = create_model(select_model)
            final = finalize_model(model)
            save_model(final, select_model+'_saved_'+datetime.date.today().strftime('%Y%m%d'))
            
            st.markdown("# 8. 予測したいデータを追加してください。")
            uploaded_file_new = st.file_uploader("CSVファイルをアップロードしてください。", type='csv', key='test')
            if uploaded_file_new is not None:
                df_new = pd.read_csv(uploaded_file_new)
                predictions = predict_model(final, data=df_new)
                predictions.to_csv(select_model+'_predict_'+datetime.date.today().strftime('%Y%m%d')+'.csv')
                st.dataframe(predictions)
                
                
        




