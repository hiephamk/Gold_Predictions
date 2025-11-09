import { useEffect, useState } from 'react';
import axios from 'axios';
import { Tabs, Box, Button, Spinner, Text, VStack,HStack, Table, Center, Grid, Heading } from '@chakra-ui/react';
import Chart from "react-apexcharts";

interface Prediction {
  date: string;
  formatted: string;
  change: number;
  confidence: number;
  open: number;
  high: number;
  low: number;
  close: number;
  price_range: number;
}

interface PredictionResponse {
  prediction: Prediction[];
  summary: {
    last_open: number;
    last_high: number;
    last_low: number;
    last_close: number;
    avg_predicted_close: number
    high_predicted: number
    low_predicted: number
    avg_range: number
    interval: string;
  };
  message: string;
}

interface FxActualPredictedDataProps {
  interval: '15min'| '30min'| '45min' | '1h'| '4h'|'1day' | '1week';
  // symbol: 'EURUSD=X' | 'JPY=X' | 'GBPUSD=X';
}

const FxPredictionResult: React.FC<FxActualPredictedDataProps> = ({ interval, }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [predictions, setPredictions] = useState<PredictionResponse | null>(null);
  // const [interval, setPredictionInterval] = useState<'15m'| '30m'| '1h'| '4h'|'1d' | '1wk'>('1h');
  // const [symbol, setSymbol] = useState<'XAU/USD' | 'EUR/USD' | 'JPY/USD' | 'GBP/USD' | 'CAD/USD' | 'AUD/USD'>('XAU/USD');

    const [summary, setSummary] = useState<PredictionResponse['summary'] | null>(null);
    const [series, setSeries] = useState<any[]>([]);

  
const handlePredict = async () => {
  setLoading(true);
  setError(null);
  setPredictions(null);

  const url = 'http://localhost:8000/api/fxprediction/result/';

  try {
    const response = await axios.post<any>(url, { interval:interval}, {
      headers: { 'Content-Type': 'application/json' },
    });

    const data = response.data;
    const preds = data.predictions;
    console.log("predicted chart data: ", preds)

    if (preds && Array.isArray(preds) && preds.length > 0) {
      setPredictions({
        prediction: preds,  // ← assign to singular for UI
        summary: data.summary,
        message: data.message || 'Prediction successful!'
      });
    } else {
      throw new Error('Invalid prediction data: predictions array missing or empty');
    }
  } catch (err: any) {
    const msg = err.response?.data?.message || err.message || 'Unknown error';
    setError(msg);
    console.error('Prediction error:', err);
  } finally {
    setLoading(false);
  }
};


useEffect(() => {
    const run = async () => {
      setLoading(true);
      try {
        await Promise.all([handlePredict()]);
      } finally {
        setLoading(false);
      }
    };

    run();                                   // immediate
    const timer = setInterval(run, 3_600_000); // hourly

    return () => clearInterval(timer);
  }, [interval]);


useEffect(() => {
    if (!predictions?.prediction?.length) return;

    const predictedCandleData = predictions.prediction.map((p) => ({
      x: new Date(p.date),
      y: [p.open, p.high, p.low, p.close, p.confidence],
    }));


    setSeries([
      {
        name: "Predicted OHLC",
        type: "candlestick",
        data: predictedCandleData,
      },
    ]);
  }, [predictions, interval]);

  const options: any = {
    chart: {
      type: "candlestick",
      height: 500,
      toolbar: { show: true },
      zoom: { enabled: false },
    },
    title: {
      text: 'XAUUSD - Predicted OHLC',
      align: "center",
      style: {
        fontSize: "18px",
        fontWeight: "bold",
      },
    },
    xaxis: {
      type: "datetime",
      labels: {
        datetimeUTC: false,
        rotate: -45,
        datetimeFormatter: {
          year: "yyyy",
          month: "MMM",
          day: "dd MMM yyyy",
          hour: "dd MMM yyyy HH:mm",
        },
      },
    },
    yaxis: {
      tooltip: { enabled: true }, 
      labels: { formatter: (val: number) => val.toFixed(2) }
    },
    plotOptions: {
      candlestick: {
        colors: {
          upward: "#00B746",
          downward: "#EF403C",
        },
      },
    },
    legend: {
      position: "top",
      horizontalAlign: "center",
    },
    tooltip: {
      shared: false,
      x: { format: "yyyy-MM-dd HH:mm" },
      custom: function ({ seriesIndex, dataPointIndex, w }: any) {
        const data = w.globals.initialSeries[seriesIndex].data[dataPointIndex];
        
        if (data.y && Array.isArray(data.y)) {
          const [open, high, low, close, confidence] = data.y;
          // const confidence = data.y
          const change = close - open;
          const changePercent = ((change / open) * 100).toFixed(2);
          
          return `
            <div style="padding: 10px; background: rgba(0,0,0,0.85); color: white; border-radius: 6px;">
              <div style="font-weight: bold; margin-bottom: 8px;">
                ${w.globals.seriesNames[seriesIndex]}
              </div>
              <div><strong>Open:</strong> $${open.toFixed(2)}</div>
              <div><strong>High:</strong> <span style="color: #00B746;">$${high.toFixed(2)}</span></div>
              <div><strong>Low:</strong> <span style="color: #EF403C;">$${low.toFixed(2)}</span></div>
              <div><strong>Close:</strong> $${close.toFixed(2)}</div>
              <div><strong>Confidence:</strong> $${confidence.toFixed(2)}</div>
              <div style="margin-top: 4px; color: ${change >= 0 ? '#00B746' : '#EF403C'};">
                <strong>Change:</strong> ${change >= 0 ? '+' : ''}$${change.toFixed(2)} (${changePercent}%)
              </div>
            </div>
          `;
        }
        return "";
      },
    },
    stroke: {
      width: [1, 1, 2],
    },
    colors: ["#008FFB", "#FEB019", "#775DD0"],
  };


  return (
    <Box w={"100%"} maxW="100%" my={"20px"}>
      <VStack gap={"10px"} align="stretch">
        
        {/* Interval Selector */}
        <HStack gap={"20px"}>

        </HStack>

        {/* Predict Button */}
        <Button
          onClick={handlePredict}
          colorScheme="blue"
          size="lg"
        >
          Run Prediction ({interval})
        </Button>

        {/* Simple Error Text */}
        {error && (
          <Box color="red.600" p={3} bg="red.50" borderRadius="md">
            {error}
          </Box>
        )}

        {/* Loading */}
        {loading && (
          <Box textAlign="center">
            <Spinner size="xl" />
            <Text mt={2}>Predicting...</Text>
          </Box>
        )}

        {/* Results */}
        {predictions && !loading && (
          <Box>
            <Box border={"1px solid"} p={"10px"} rounded={"7px"} shadow="3px 3px 15px 5px rgb(75, 75, 79)">
              <Center>
                <Heading fontWeight="semibold">
                  Last Actual Price ({predictions.summary.interval})
                </Heading>
              </Center>
              <HStack justifyContent={"space-around"}>
                <Box>
                  <Text>Last Open: ${predictions.summary.last_open}</Text>
                  <Text>Last High: ${predictions.summary.last_high}</Text>
                  <Text>Last Low: ${predictions.summary.last_low}</Text>
                  <Text>Last Close: ${predictions.summary.last_close}</Text>
                </Box>
                <Box>
                  <HStack><Text>Average Predicted Close:</Text><Text>${predictions.summary.avg_predicted_close}</Text></HStack>
                  <HStack><Text>Average Predicted High:</Text><Text>${predictions.summary.high_predicted}</Text></HStack>
                  <HStack><Text>Average Predicted Low:</Text><Text>${predictions.summary.low_predicted}</Text></HStack>
                  <HStack><Text>Average Predicted Range: </Text><Text>${predictions.summary.avg_range}</Text></HStack>
                </Box>
              </HStack>
            </Box>
              <Box w={"100%"} justifyContent={"space-between"} mt={"10px"}>
                  
                  
              </Box>
              <Tabs.Root defaultValue={"chart"} shadow="3px 3px 15px 5px rgb(75, 75, 79)" rounded={"7px"}>
                <Tabs.List rounded={"5px"}>
                  <Tabs.Trigger value='chart'>Chart</Tabs.Trigger>
                  <Tabs.Trigger value='table'>Table Data</Tabs.Trigger>
                </Tabs.List>
                <Tabs.Content value='table'>
                  <Box rounded={"5px"} p={"5px"}>
                    <Center fontSize={"24px"} fontWeight={"bold"}>
                        Gold Price Predictions For The Next {predictions.prediction.length} Time Intervals
                    </Center>
                    <Table.Root showColumnBorder>
                      <Table.Header  borderTop={"1px solid"}>
                        <Table.Row>
                          <Table.ColumnHeader fontWeight={"bold"} fontSize={"18px"}>Date</Table.ColumnHeader>
                          <Table.ColumnHeader fontWeight={"bold"} fontSize={"18px"}>Open</Table.ColumnHeader>
                          <Table.ColumnHeader fontWeight={"bold"} fontSize={"18px"}>High</Table.ColumnHeader>
                          <Table.ColumnHeader fontWeight={"bold"} fontSize={"18px"}>Low</Table.ColumnHeader>
                          <Table.ColumnHeader fontWeight={"bold"} fontSize={"18px"}>Close</Table.ColumnHeader>
                          <Table.ColumnHeader fontWeight={"bold"} fontSize={"18px"}>Change</Table.ColumnHeader>
                          <Table.ColumnHeader fontWeight={"bold"} fontSize={"18px"}>Range</Table.ColumnHeader>
                          <Table.ColumnHeader fontWeight={"bold"} fontSize={"18px"}>Confidence</Table.ColumnHeader>
                        </Table.Row>
                      </Table.Header>
                      <Table.Body>
                        {
                          predictions.prediction.map((data, idx) => (
                            <Table.Row key={idx}>
                              <Table.Cell>{data.formatted}</Table.Cell>
                              <Table.Cell>{data.open}</Table.Cell>
                              <Table.Cell>{data.high}</Table.Cell>
                              <Table.Cell>{data.low}</Table.Cell>
                              <Table.Cell>{data.close}</Table.Cell>
                              <Table.Cell>{data.change}</Table.Cell>
                              <Table.Cell>{data.price_range}</Table.Cell>
                              <Table.Cell>{data.confidence}%</Table.Cell>
                            </Table.Row>
                          ))
                        }
                      </Table.Body>
                    </Table.Root>
                  </Box>
                </Tabs.Content>
                <Tabs.Content value='chart'>
                  <Box my={"10px"}>
                    <Center fontSize={"24px"} fontWeight={"bold"}>Gold Price Predictions For The Next {predictions.prediction.length} Time Intervals </Center>
                    {series.length > 0 && !loading && (
                      <Box borderTop={"1px solid"} w={"100%"}>
                        <Chart options={options} series={series} type="candlestick" height={500} />
                      </Box>
                    )}
                  </Box>
                </Tabs.Content>
              </Tabs.Root>
          </Box>
        )}
      </VStack>
    </Box>
  );
};

export default FxPredictionResult;
