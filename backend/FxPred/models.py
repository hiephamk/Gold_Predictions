from django.db import models

class PredictedDataBase(models.Model):
    """Abstract base model for predicted data across different intervals"""
    date = models.DateTimeField(unique=True, db_index=True)
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    price_range = models.FloatField()  # Renamed from 'range'
    change = models.FloatField()
    confidence = models.FloatField()

    def __str__(self):
        return f"{self.date} - {self.close:.2f}"

    class Meta:
        abstract = True
        ordering = ['-date']


class PredictedData_15m(PredictedDataBase):
    class Meta(PredictedDataBase.Meta):
        db_table = 'predicted_data_15m'  # Optional: explicit table name


class PredictedData_30m(PredictedDataBase):
    class Meta(PredictedDataBase.Meta):
        db_table = 'predicted_data_30m'


class PredictedData_1h(PredictedDataBase):
    class Meta(PredictedDataBase.Meta):
        db_table = 'predicted_data_1h'


class PredictedData_4h(PredictedDataBase):
    class Meta(PredictedDataBase.Meta):
        db_table = 'predicted_data_4h'


class PredictedData_1d(PredictedDataBase):
    class Meta(PredictedDataBase.Meta):
        db_table = 'predicted_data_1d'


class PredictedData_1wk(PredictedDataBase):
    class Meta(PredictedDataBase.Meta):
        db_table = 'predicted_data_1wk'
