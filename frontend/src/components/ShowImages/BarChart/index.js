import { BarChart } from '@mui/x-charts/BarChart';

export default function SimpleCharts() {
  return (
    <BarChart
      xAxis={[
        {
          id: "barCategories",
          data: ["bar A", "bar B", "bar C", "bar D"],
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
          data: [2, 5, 3, 10],
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
