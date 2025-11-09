from .models import *
def save_actual_gold_prices(df):
    objects = []
    for _, row in df.inerrows():
        ts = row['Date']
        price = row['Close']

        objects.append(ActualGoldPrice(timestamp=ts, actual_closed_price=price))

    ActualGoldPrice.objects.bulk_create(
        objects,
        ignore_conflicts=True
    )