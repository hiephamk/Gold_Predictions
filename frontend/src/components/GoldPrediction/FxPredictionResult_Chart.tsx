import { useEffect, useState } from 'react';
import axios from 'axios';
import { Box, Button, Spinner, Text, VStack, HStack, Badge, Grid } from '@chakra-ui/react';
import Chart from "react-apexcharts";

interface OHLCPrediction {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface FxActualPredictedDataProps {
  interval: '15min'| '30min'| '45min' | '1h'| '4h'|'1day' | '1week';
}

const FxPredictionResult_Chart: React.FC<FxActualPredictedDataProps> = ({ interval }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [series, setSeries] = useState<any[]>([]);
  const [predictions, setPredictions] = useState<OHLCPrediction[]>([]);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const toISO = (dateStr: string): string =>
    dateStr.includes("T") ? dateStr.trim() : dateStr.trim().replace(" ", "T");
  
  const fetch_predicted_prices = async () => {
    // const url = `http://localhost:8000/api/fxprediction/predicted-prices-${interval}/`;
    const url = `${import.meta.env.VITE_PREDICTION_RESULT_URL}-${interval}/`
    
    try {
      const { data } = await axios.get(url);
      
      if (!data || data.length === 0) {
        throw new Error("No prediction data available");
      }
      
      const normalized: OHLCPrediction[] = (data ?? []).map((p: any) => ({
        date: toISO(p.date),
        open: Number(p.open),
        high: Number(p.high),
        low: Number(p.low),
        close: Number(p.close),
      }));
      
      setPredictions(normalized);
      setLastUpdate(new Date());
      return data;
    } catch (err: any) {
      console.error("API Error:", err);
      throw new Error(err.response?.data?.message || "Failed to load predictions");
    }
  };

  /* -------------------------- Load Data -------------------------- */
  useEffect(() => {
    let isMounted = true;
    let timer: NodeJS.Timeout;

    const load = async () => {
      if (!isMounted) return;
      
      setPredictions([]);
      setError(null);
      setLoading(true);

      try {
        await fetch_predicted_prices();
      } catch (err: any) {
        if (isMounted) setError(err.message || "Failed to load data");
      } finally {
        if (isMounted) setLoading(false);
      }
    };

    // Load immediately
    load();

    // Refresh every hour
    timer = setInterval(() => {
      load();
    }, 3_600_000); // 1 hour

    // Cleanup on unmount OR interval change
    return () => {
      isMounted = false;
      clearInterval(timer);
    };
  }, [interval]);

  /* -------------------------- Build Chart Series -------------------------- */
  useEffect(() => {
    if (!predictions?.length) {
      setSeries([]);
      return;
    }

    const predictedCandleData = predictions.map((p) => ({
      x: new Date(p.date).getTime(),
      y: [p.open, p.high, p.low, p.close],
    }));

    setSeries([
      {
        name: "Predicted OHLC",
        type: "candlestick",
        data: predictedCandleData,
      },
    ]);
  }, [predictions]);

  /* -------------------------- Calculate Stats -------------------------- */
  const stats = predictions.length > 0 ? {
    avgClose: (predictions.reduce((sum, p) => sum + p.close, 0) / predictions.length).toFixed(2),
    highestHigh: Math.max(...predictions.map(p => p.high)).toFixed(2),
    lowestLow: Math.min(...predictions.map(p => p.low)).toFixed(2),
    trend: predictions[predictions.length - 1]?.close > predictions[0]?.close ? 'increase' : 'decrease',
    changePercent: predictions.length >= 2 
      ? (((predictions[predictions.length - 1].close - predictions[0].close) / predictions[0].close) * 100).toFixed(2)
      : '0.00'
  } : null;

  /* -------------------------- Chart Options -------------------------- */
  const options: any = {
    chart: {
      type: "candlestick",
      height: 600,
      toolbar: { 
        show: true,
        tools: {
          download: true,
          selection: true,
          zoom: true,
          zoomin: true,
          zoomout: true,
          pan: true,
          reset: true
        }
      },
      zoom: { enabled: true },
      background: 'transparent',
    },
    title: {
      text: `XAUUSD - Predicted OHLC (${interval})`,
      align: "center",
      style: {
        fontSize: "20px",
        fontWeight: "600",
        color: '#2D3748'
      },
    },
    xaxis: {
      type: "datetime",
      labels: {
        datetimeUTC: false,
        rotate: -45,
        style: {
          fontSize: '12px',
        },
        datetimeFormatter: {
          year: "yyyy",
          month: "MMM 'yy",
          day: "dd MMM",
          hour: "HH:mm",
        },
      },
      axisBorder: {
        show: true,
        color: '#E2E8F0'
      },
      axisTicks: {
        show: true,
        color: '#E2E8F0'
      }
    },
    yaxis: {
      tooltip: { enabled: true },
      labels: { 
        formatter: (val: number) => `$${val.toFixed(2)}`,
        style: {
          fontSize: '12px',
        }
      },
      axisBorder: {
        show: true,
        color: '#E2E8F0'
      }
    },
    plotOptions: {
      candlestick: {
        colors: {
          upward: "#10B981",
          downward: "#EF4444",
        },
        wick: {
          useFillColor: true,
        }
      },
    },
    grid: {
      borderColor: '#E2E8F0',
      strokeDashArray: 1,
      xaxis: {
        lines: {
          show: true
        }
      },
      yaxis: {
        lines: {
          show: true
        }
      },
    },
    tooltip: {
      theme: 'dark',
      shared: false,
      x: { 
        format: "dd MMM yyyy HH:mm" 
      },
      custom: function ({ seriesIndex, dataPointIndex, w }: any) {
        const data = w.globals.initialSeries[seriesIndex].data[dataPointIndex];
        
        if (data.y && Array.isArray(data.y)) {
          const [open, high, low, close] = data.y;
          const change = close - open;
          const changePercent = ((change / open) * 100).toFixed(2);
          const date = new Date(data.x).toLocaleString();
          
          return `
            <div style="padding: 12px; background: rgba(0,0,0,0.9); color: white; border-radius: 8px; min-width: 220px;">
              <div style="font-weight: 600; margin-bottom: 10px; font-size: 13px; border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 6px;">
                ${date}
              </div>
              <div style="display: grid; gap: 4px; font-size: 12px;">
                <div style="display: flex; justify-content: space-between;">
                  <span style="color: #94A3B8;">Open:</span>
                  <span style="font-weight: 500;">$${open.toFixed(2)}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                  <span style="color: #94A3B8;">High:</span>
                  <span style="color: #10B981; font-weight: 500;">$${high.toFixed(2)}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                  <span style="color: #94A3B8;">Low:</span>
                  <span style="color: #EF4444; font-weight: 500;">$${low.toFixed(2)}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                  <span style="color: #94A3B8;">Close:</span>
                  <span style="font-weight: 500;">$${close.toFixed(2)}</span>
                </div>
                <div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.2); display: flex; justify-content: space-between;">
                  <span style="color: #94A3B8;">Change:</span>
                  <span style="color: ${change >= 0 ? '#10B981' : '#EF4444'}; font-weight: 600;">
                    ${change >= 0 ? '+' : ''}$${change.toFixed(2)} (${changePercent}%)
                  </span>
                </div>
              </div>
            </div>
          `;
        }
        return "";
      },
    },
    stroke: {
      width: 1,
    },
  };

  return (
    <Box w="100%" maxW="100%" my="10px">
      <VStack gap="20px" align="stretch">
        {/* Header with Stats */}
        {stats && !loading && (
          <Grid templateColumns={{ base: "repeat(2, 1fr)", md: "repeat(4, 1fr)" }} gap={4}>
            <Box p={4} borderRadius="lg" border="1px solid" borderColor="gray.200">
              <Text fontSize="sm" color="gray.600" mb={2}>Avg Price</Text>
              <Text fontSize="2xl" fontWeight="bold">${stats.avgClose}</Text>
            </Box>
            
            <Box p={4} borderRadius="lg" border="1px solid" borderColor="gray.200">
              <Text fontSize="sm" color="gray.600" mb={2}>Highest</Text>
              <Text fontSize="2xl" fontWeight="bold" color="green.500">${stats.highestHigh}</Text>
            </Box>
            
            <Box p={4} borderRadius="lg" border="1px solid" borderColor="gray.200">
              <Text fontSize="sm" color="gray.600" mb={2}>Lowest</Text>
              <Text fontSize="2xl" fontWeight="bold" color="red.500">${stats.lowestLow}</Text>
            </Box>
            
            <Box p={4} borderRadius="lg" border="1px solid" borderColor="gray.200">
              <Text fontSize="sm" color="gray.600" mb={2}>Trend</Text>
              <Text fontSize="2xl" fontWeight="bold" color={stats.trend === 'increase' ? 'green.500' : 'red.500'}>
                {stats.trend === 'increase' ? '↗' : '↘'} {stats.changePercent}%
              </Text>
            </Box>
          </Grid>
        )}

        {/* Last Update Info */}
        {lastUpdate && !loading && (
          <HStack justify="space-between" px={2}>
            <Text fontSize="sm" color="gray.600">
              Last updated: {lastUpdate.toLocaleString()}
            </Text>
            <Badge colorScheme="blue" fontSize="sm">
              {predictions.length} predictions
            </Badge>
          </HStack>
        )}

        {/* Error Message */}
        {error && (
          <Box 
            p={4} 
            bg="red.50" 
            borderRadius="lg" 
            border="1px solid" 
            borderColor="red.200"
            color="red.700"
          >
            <Text fontWeight="semibold">⚠️ Error loading predictions</Text>
            <Text fontSize="sm" mt={1}>{error}</Text>
          </Box>
        )}

        {/* Loading State */}
        {loading && (
          <Box 
            textAlign="center" 
            py={20} 
           
            borderRadius="lg" 
            border="1px solid" 
          >
            <Spinner size="xl" color="blue.500" thickness="4px" />
            <Text mt={4} color="gray.600" fontWeight="medium">
              Loading predictions for {interval}...
            </Text>
          </Box>
        )}

        {/* Chart */}
        {series.length > 0 && !loading && (
          <Box p={"10px"}>
            <Chart 
              options={options} 
              series={series} 
              type="candlestick" 
              height={350} 
            />
          </Box>
        )}

        {/* Empty State */}
        {!loading && !error && predictions.length === 0 && (
          <Box 
            textAlign="center" 
            py={20} 
            borderRadius="lg"
            border="1px dashed"
            borderColor="gray.300"
          >
            <Text fontSize="lg" color="gray.600">
              No prediction data available
            </Text>
            <Text fontSize="sm" color="gray.500" mt={2}>
              Try selecting a different time interval
            </Text>
          </Box>
        )}
      </VStack>
    </Box>
  );
};

export default FxPredictionResult_Chart;


// import { useEffect, useState } from 'react';
// import axios from 'axios';
// import { Box, Button, Spinner, Text, VStack } from '@chakra-ui/react';
// import Chart from "react-apexcharts";

// interface Prediction {
//   date: string;
//   formatted: string;
//   change: number;
//   confidence: number;
//   open: number;
//   high: number;
//   low: number;
//   close: number;
//   price_range: number;
// }
// interface OHLCPrediction {
//   date: string;
//   open: number;
//   high: number;
//   low: number;
//   close: number;
// }
// interface PredictionResponse {
//   prediction: Prediction[];
//   summary: {
//     last_open: number;
//     last_high: number;
//     last_low: number;
//     last_close: number;
//     avg_predicted_close: number
//     high_predicted: number
//     low_predicted: number
//     avg_range: number
//     interval: string;
//   };
//   message: string;
// }

// interface FxActualPredictedDataProps {
//   interval: '15min'| '30min'| '45min' | '1h'| '4h'|'1day' | '1week';
//   // symbol: 'EURUSD=X' | 'JPY=X' | 'GBPUSD=X';
// }

// const FxPredictionResult_Chart: React.FC<FxActualPredictedDataProps> = ({ interval, }) => {
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState<string | null>(null);
  
//   const [summary, setSummary] = useState<PredictionResponse['summary'] | null>(null);
//   const [series, setSeries] = useState<any[]>([]);

//   const [predictions, setPredictions] = useState<OHLCPrediction[]>([]);

//   const toISO = (dateStr: string): string =>
//     dateStr.includes("T") ? dateStr.trim() : dateStr.trim().replace(" ", "T");
  
//   const fetch_predicted_prices = async () => {
//     const url = `http://localhost:8000/api/fxprediction/predicted-prices-${interval}/`;
//     console.log("Fetching from:", url); // Add this
    
//     try {
//       const { data } = await axios.get(url);
//       console.log("Raw API response:", data); // Add this
      
//       const normalized: OHLCPrediction[] = (data ?? []).map((p: any) => ({
//         date: toISO(p.date),
//         open: Number(p.open),
//         high: Number(p.high),
//         low: Number(p.low),
//         close: Number(p.close),
//       }));
      
//       console.log("data candle:", normalized);
//       setPredictions(normalized);
//       return data;
//     } catch (err: any) {
//       console.error("API Error:", err); // Add this to see the actual error
//       setError("Failed to load predictions");
//     }
//   };
//   /* -------------------------- Load Data -------------------------- */
//   useEffect(() => {
//   let isMounted = true;
//   let timer: NodeJS.Timeout;

//   const load = async () => {
//     if (!isMounted) return;

//     // Optional: clear old data on interval change
//     setPredictions([]);
//     setError(null);
//     setLoading(true);

//     try {
//       await Promise.all([
//         fetch_predicted_prices()
//       ]);
//     } catch (err: any) {
//       if (isMounted) setError(err.message || "Failed to load data");
//     } finally {
//       if (isMounted) setLoading(false);
//     }
//   };

//   // Load immediately
//   load();

//   // Refresh every hour
//   timer = setInterval(() => {
//     load();
//   }, 3_600_000); // 1 hour

//   // Cleanup on unmount OR interval change
//   return () => {
//     isMounted = false;
//     clearInterval(timer);
//   };
// }, [interval]);

// // const handlePredict = async () => {
// //   setLoading(true);
// //   setError(null);
// //   setPredictions(null);

// //   const url = 'http://localhost:8000/api/fxprediction/result/';

// //   try {
// //     const response = await axios.post<any>(url, { interval:interval}, {
// //       headers: { 'Content-Type': 'application/json' },
// //     });

// //     const data = response.data;
// //     const preds = data.predictions;
// //     console.log("predicted chart data: ", preds)

// //     if (preds && Array.isArray(preds) && preds.length > 0) {
// //       setPredictions({
// //         prediction: preds,  // ← assign to singular for UI
// //         summary: data.summary,
// //         message: data.message || 'Prediction successful!'
// //       });
// //     } else {
// //       throw new Error('Invalid prediction data: predictions array missing or empty');
// //     }
// //   } catch (err: any) {
// //     const msg = err.response?.data?.message || err.message || 'Unknown error';
// //     setError(msg);
// //     console.error('Prediction error:', err);
// //   } finally {
// //     setLoading(false);
// //   }
// // };


// // useEffect(() => {
// //     const run = async () => {
// //       setLoading(true);
// //       try {
// //         await Promise.all([handlePredict()]);
// //       } finally {
// //         setLoading(false);
// //       }
// //     };

// //     run();                                   // immediate
// //     const timer = setInterval(run, 3_600_000); // hourly

// //     return () => clearInterval(timer);
// //   }, [interval]);

// useEffect(()=>{
//   console.log("predicted data candle: ", predictions)
// }, [predictions])

// useEffect(() => {
//     if (!predictions?.length) return;

//     const predictedCandleData = predictions.map((p) => ({
//       x: new Date(p.date),
//       y: [p.open, p.high, p.low, p.close],
//     }));


//     setSeries([
//       {
//         name: "Predicted OHLC",
//         type: "candlestick",
//         data: predictedCandleData,
//       },
//     ]);
//   }, [predictions, interval]);

//   const options: any = {
//     chart: {
//       type: "candlestick",
//       height: 500,
//       toolbar: { show: true },
//       zoom: { enabled: false },
//     },
//     title: {
//       text: 'XAUUSD - Predicted OHLC',
//       align: "center",
//       style: {
//         fontSize: "18px",
//         fontWeight: "bold",
//       },
//     },
//     xaxis: {
//       type: "datetime",
//       labels: {
//         datetimeUTC: false,
//         rotate: -45,
//         datetimeFormatter: {
//           year: "yyyy",
//           month: "MMM",
//           day: "dd MMM yyyy",
//           hour: "dd MMM yyyy HH:mm",
//         },
//       },
//     },
//     yaxis: {
//       tooltip: { enabled: true }, 
//       labels: { formatter: (val: number) => val.toFixed(2) }
//     },
//     plotOptions: {
//       candlestick: {
//         colors: {
//           upward: "#00B746",
//           downward: "#EF403C",
//         },
//       },
//     },
//     legend: {
//       position: "top",
//       horizontalAlign: "center",
//     },
//     tooltip: {
//       shared: false,
//       x: { format: "yyyy-MM-dd HH:mm" },
//       custom: function ({ seriesIndex, dataPointIndex, w }: any) {
//         const data = w.globals.initialSeries[seriesIndex].data[dataPointIndex];
        
//         if (data.y && Array.isArray(data.y)) {
//           const [open, high, low, close] = data.y;
//           const change = close - open;
//           const changePercent = ((change / open) * 100).toFixed(2);
          
//           return `
//             <div style="padding: 10px; background: rgba(0,0,0,0.85); color: white; border-radius: 6px;">
//               <div style="font-weight: bold; margin-bottom: 8px;">
//                 ${w.globals.seriesNames[seriesIndex]}
//               </div>
//               <div><strong>Open:</strong> $${open.toFixed(2)}</div>
//               <div><strong>High:</strong> <span style="color: #00B746;">$${high.toFixed(2)}</span></div>
//               <div><strong>Low:</strong> <span style="color: #EF403C;">$${low.toFixed(2)}</span></div>
//               <div><strong>Close:</strong> $${close.toFixed(2)}</div>
//               <div style="margin-top: 4px; color: ${change >= 0 ? '#00B746' : '#EF403C'};">
//                 <strong>Change:</strong> ${change >= 0 ? '+' : ''}$${change.toFixed(2)} (${changePercent}%)
//               </div>
//             </div>
//           `;
//         }
//         return "";
//       },
//     },
//     stroke: {
//       width: [1, 1, 2],
//     },
//     colors: ["#008FFB", "#FEB019", "#775DD0"],
//   };


//   return (
//     <Box w={"100%"} maxW="100%" my={"20px"}>
//       <VStack gap={"10px"} align="stretch">
//         {/* <Button
//           onClick={handlePredict}
//           colorScheme="blue"
//           size="lg"
//         >
//           Run Prediction ({interval})
//         </Button> */}
//         {error && (
//           <Box color="red.600" bg="red.50" borderRadius="md">
//             {error}
//           </Box>
//         )}
//         {loading && (
//           <Box textAlign="center">
//             <Spinner size="xl" />
//             <Text mt={2}>Predicting...</Text>
//           </Box>
//         )}
//         {predictions && !loading && (
//           <Box>
//             <Box my={"10px"}>
//               {series.length > 0 && !loading && (
//                 <Box rounded={"7px"} border={"1px solid"} w={"100%"}>
//                   <Chart options={options} series={series} type="candlestick" height={500} />
//                 </Box>
//               )}
//             </Box>
//           </Box>
//         )}
//       </VStack>
//     </Box>
//   );
// };

// export default FxPredictionResult_Chart;
