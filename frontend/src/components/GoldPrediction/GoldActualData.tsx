import { Box, Table, Spinner, Text, Alert, Heading, Center } from '@chakra-ui/react';
import { useState, useEffect } from 'react';
import axios from 'axios';
import formatDate from '../formatDate';
import Chart from "react-apexcharts";

interface ActualPrice {
  Date: string;
  Open: number;
  High: number;
  Low: number;
  Close: number;
}

interface Props {
  interval: '15m' | '30m' | '1h' | '4h' | '1d' | '1wk';
}

const toISO = (dateStr: string): string =>
  dateStr.includes('T') ? dateStr.trim() : dateStr.trim().replace(' ', 'T');

const GoldActualData: React.FC<Props> = ({ interval }) => {
  const [actualPrice, setActualPrice] = useState<ActualPrice[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [series, setSeries] = useState<any[]>([]);

  const fetchActual = async () => {
    try {
      const { data } = await axios.post(
        'http://localhost:8000/api/fxprediction/actual-prices/',
        { interval },
        { headers: { 'Content-Type': 'application/json' } }
      );

      const raw: any[] = data.data ?? [];

      if (!Array.isArray(raw) || raw.length === 0) {
        setActualPrice([]);
        return;
      }

      const parsed: ActualPrice[] = raw
        .map((row) => ({
          Date: toISO(String(row.Date)),
          Open: Number(row.Open),
          High: Number(row.High),
          Low: Number(row.Low),
          Close: Number(row.Close),
        }))
        .filter(
          (d): d is ActualPrice =>
            !isNaN(new Date(d.Date).getTime()) &&
            !isNaN(d.Open) &&
            !isNaN(d.High) &&
            !isNaN(d.Low) &&
            !isNaN(d.Close)
        );

      setActualPrice(parsed);
    } catch (err: any) {
      const msg =
        err.response?.data?.error || err.message || 'Unknown error';
      setError(`Failed to load actual prices: ${msg}`);
    }finally {
      setLoading(false)
    }
  };

  useEffect(() => {
    const run = async () => {
      setLoading(true);
      try {
        await Promise.all([fetchActual()]);
      } finally {
        setLoading(false);
      }
    };

    run();
    const timer = setInterval(run, 3_600_000);

    return () => clearInterval(timer);
  }, [interval]);

  useEffect(() => {
    console.log('actualPrice (hook):', actualPrice);
  }, [actualPrice]);

  //candlestick of actual gold price
  useEffect(() => {
    if (!actualPrice?.length) return;
    const gold_actual_candlestick_data = actualPrice
    .slice(0,150)
    .map((d) => ({
      x: new Date(d.Date),
      y: [d.Open, d.High, d.Low, d.Close]
    }))
    setSeries([
      {
        name: "Actual OHLC",
        type: "candlestick",
        data: gold_actual_candlestick_data,
      }
    ])
  },[actualPrice,interval])
  
  const options: any = {
    chart: {
      type: "candlestick",
      height: 500,
      toolbar: {show: true},
      zoom: { enable: false}
    },
    title: {
      text: `XAUUSD - Actual Price - ${interval} (source: Yahoo Finance)`,
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
      tooltip: { 
        enabled: true,
      },
      labels: { formatter: (val: number) => val.toFixed(2)},
      min: undefined,
      max: undefined,
      forceNiceScale: true,
      tickAmount: 30,
    },
    plotOptions: {
      candlestick: {
        color: {
          upward: "#00B746",
          downward: "#EF403C",
        },
      },
    },
    legend: {
      position: "top",
      horizontalAlign: "center",
    },
    grid: {

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
        enabled: true,
        shared: false,
        x: { format: "yyyy-MM-dd HH:mm"},
        custom: function ({ seriesIndex, dataPointIndex, w}: any){
          const data = w.globals.initialSeries[seriesIndex].data[dataPointIndex];
          if (data.y && Array.isArray(data.y)){
            const [open, high, low, close] = data.y;
            return `
              <div style="padding: 10px; background: rgba(0,0,0,0.85); color: white; border-radius: 6px;">
              <div style="font-weight: bold; margin-bottom: 8px;">
                ${w.globals.seriesNames[seriesIndex]}
              </div>
              <div><strong>Open:</strong> $${open.toFixed(2)}</div>
              <div><strong>High:</strong> <span style="color: #00B746;">$${high.toFixed(2)}</span></div>
              <div><strong>Low:</strong> <span style="color: #EF403C;">$${low.toFixed(2)}</span></div>
              <div><strong>Close:</strong> $${close.toFixed(2)}</div>
            </div>
          `;
          }
          return ""
        },
      },
    stroke: {
      width: 1,
    },
    colors: ["#008FFB", "#FEB019", "#775DD0"],
  }

  return (
    <Box w="100%" rounded="md" p={"10px"} overflowX="auto">
      <Box my={"10px"}>
        {series.length > 0 && !loading && (
          <Box  w={"100%"} pr={"30px"}>
            <Chart options={options} series={series} type="candlestick" height={"350px"}/>
          </Box>
        )}
      </Box>
      {/* <Box>      
        <Center my={"20px"}><Heading fontSize={"24px"}>Actual Gold Price from yfinance</Heading></Center>
            {loading && (
              <Box textAlign="center" py={8}>
                <Spinner size="lg" color="blue.500" />
                <Text ml={3} display="inline">
                  Loading {interval} data…
                </Text>
              </Box>
            )}
            {!loading && !error && actualPrice.length > 0 && (
              <Table.Root size="sm" showColumnBorder>
                <Table.Header>
                  <Table.Row>
                    <Table.ColumnHeader>Date</Table.ColumnHeader>
                    <Table.ColumnHeader>Open</Table.ColumnHeader>
                    <Table.ColumnHeader>High</Table.ColumnHeader>
                    <Table.ColumnHeader>Low</Table.ColumnHeader>
                    <Table.ColumnHeader>Close</Table.ColumnHeader>
                  </Table.Row>
                </Table.Header>

                <Table.Body>
                  {actualPrice
                  .slice(0,5)
                  .map((row, idx) => (
                    <Table.Row key={idx}>
                      <Table.Cell>{formatDate(row.Date)}</Table.Cell>
                      <Table.Cell>{row.Open.toFixed(2)}</Table.Cell>
                      <Table.Cell>{row.High.toFixed(2)}</Table.Cell>
                      <Table.Cell>{row.Low.toFixed(2)}</Table.Cell>
                      <Table.Cell>{row.Close.toFixed(2)}</Table.Cell>
                    </Table.Row>
                  ))}
                </Table.Body>
              </Table.Root>
            )}

            {!loading && !error && actualPrice.length === 0 && (
              <Text textAlign="center" color="gray.500" py={8}>
                No data available for <strong>{interval}</strong>.
              </Text>
            )}
      </Box> */}
    </Box>
  );
};

export default GoldActualData;