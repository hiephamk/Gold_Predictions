from django.urls import path
from . import views


urlpatterns = [
    path('fxprediction/result/', views.predict_future_prices, name='fx_predict'),
    path('fxprediction/actual-prices/', views.fetch_actual_data, name='fx_predict_history'),

    path('fxprediction/predicted-prices-15m/', views.PredictedData_15m_View, name='fx_predicted_data_15m'),
    path('fxprediction/predicted-prices-30m/', views.PredictedData_30m_View, name='fx_predicted_data_30m'),
    path('fxprediction/predicted-prices-1h/', views.PredictedData_1h_View, name='fx_predicted_data_1h'),
    path('fxprediction/predicted-prices-4h/', views.PredictedData_4h_View, name='fx_predicted_data_4h'),
    path('fxprediction/predicted-prices-1d/', views.PredictedData_1d_View, name='fx_predicted_data_1d'),
    path('fxprediction/predicted-prices-1wk/', views.PredictedData_1wk_View, name='fx_predicted_data_1wk'),
]
