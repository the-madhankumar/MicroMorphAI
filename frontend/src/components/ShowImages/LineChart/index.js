import { LineChart } from '@mui/x-charts/LineChart';

export default function BasicLineChart({ xdata, ydata }) {
  return (
    <LineChart
      xAxis={[
        {
          data: xdata,
          labelStyle: { fill: "#FFFFFF" },
          tickLabelStyle: { fill: "#FFFFFF" },
        },
      ]}
      yAxis={[
        {
          labelStyle: { fill: "#FFFFFF" },
          tickLabelStyle: { fill: "#FFFFFF" },
        },
      ]}
      series={[
        {
          data: ydata,
          color: "#FFFFFF",   
        },
      ]}
      sx={{
        "& .MuiChartsLegend-root text": { fill: "#FFFFFF" },
        "& .MuiChartsAxis-line": { stroke: "#FFFFFF" },
        "& .MuiChartsAxis-tick": { stroke: "#FFFFFF" },
      }}
      height={300}
    />
  );
}
