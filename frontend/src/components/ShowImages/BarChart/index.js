import { BarChart } from '@mui/x-charts/BarChart';

export default function SimpleCharts({xdataClass, ydataCount}) {
  return (
    <BarChart
      xAxis={[
        {
          id: "barCategories",
          data: xdataClass,
          labelStyle: { fill: "#FFFFFF" }, 
          tickLabelStyle: { fill: "#FFFFFF" }
        }
      ]}
      yAxis={[
        {
          tickLabelStyle: { fill: "#FFFFFF" }
        }
      ]}
      series={[
        {
          data: ydataCount,
          color: "#006BD6"
        }
      ]}
      height={380}
      width={500}
      sx={{
        "& .MuiChartsAxis-label": { fill: "#FFFFFF" },
        "& .MuiChartsAxis-tickLabel": { fill: "#FFFFFF" }
      }}
    />
  );
}
