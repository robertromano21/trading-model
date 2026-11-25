using System;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Text.Json.Nodes;
using ClosedXML.Excel;

namespace ADXUpdater
{
    public class Program
    {
        static void Main(string[] args)
        {
            string[] files =
                Directory.GetFiles(
                    @"C:\Massive",
                    "prices_*.json");

            Array.Sort(files);

            foreach (string file in files)
            {
                string fileName =
                    Path.GetFileNameWithoutExtension(file);

                string datePart =
                    fileName.Replace("prices_", "");

                DateTime marketDate =
                    DateTime.ParseExact(
                        datePart,
                        "yyyy-MM-dd",
                        null);

                Console.WriteLine(
                    $"Processing {fileName}");

                UpdateWorkbook(
                    file,
                    marketDate);
                string processedFolder =
    @"C:\Massive\Processed";

                Directory.CreateDirectory(
                    processedFolder);

                string destinationFile =
                    Path.Combine(
                        processedFolder,
                        Path.GetFileName(file));

                File.Move(
                    file,
                    destinationFile,
                    true);

                Console.WriteLine(
                    $"Archived: {Path.GetFileName(file)}");
            }
        }

        public static void UpdateWorkbook(
            string jsonPath,
    DateTime marketDate)
{
            string filePath =
                @"C:\Users\antro\OneDrive\Documents\Stocks\PAPERTRADE1\ADX worksheet 2.0.xlsm";
            Console.WriteLine("Workbook Path:");
            Console.WriteLine(filePath);

            Console.WriteLine("Exists:");
            Console.WriteLine(File.Exists(filePath));



            string jsonText = File.ReadAllText(jsonPath);

            JsonNode root = JsonNode.Parse(jsonText);

            JsonArray results = root["results"].AsArray();
            string[] tickers =
{
    "NVDA",
    "ARM",
    "TSLA",
    "BROS",
    "CHWY",
    "FLY",
    "MRNA",
    "STLD",
    "HUT",
    "PLTR"
};
            Console.WriteLine("Opening workbook:");
            Console.WriteLine(filePath);

            Console.WriteLine("File Exists: " + File.Exists(filePath));


            using (var workbook = new XLWorkbook(filePath))
            {
                foreach (string ticker in tickers)
                {
                    JsonObject stockRecord = null;

                    foreach (JsonObject stock in results)
                    {
                        if (stock["T"]?.ToString() == ticker)
                        {
                            stockRecord = stock;
                            break;
                        }
                    }

                    if (stockRecord == null)
                    {
                        Console.WriteLine($"{ticker} not found.");
                        continue;
                    }

                    double open = stockRecord["o"].GetValue<double>();
                    double high = stockRecord["h"].GetValue<double>();
                    double low = stockRecord["l"].GetValue<double>();
                    double close = stockRecord["c"].GetValue<double>();
                    double volume = stockRecord["v"].GetValue<double>();

                    Console.WriteLine(
    $"{ticker} O:{open} H:{high} L:{low} C:{close} V:{volume}"
);

                    Console.WriteLine($"Updating {ticker}");

                    var worksheet = workbook.Worksheet(ticker.ToLower());

                    bool dateExists = false;
                    int newRow = 2;

                    while (!worksheet.Cell(newRow, 1).IsEmpty())
                    {
                        var cellValue =
    worksheet.Cell(newRow, 1).Value.ToString();

                        DateTime existingDate;

                        if (DateTime.TryParse(cellValue, out existingDate))
                        {
                            if (existingDate.Date == marketDate.Date)
                            {
                                dateExists = true;
                                break;
                            }
                        }

                        newRow++;
                    }

                    if (dateExists)
                    {
                        Console.WriteLine(
                            $"{ticker} already contains {marketDate:d}. Skipping.");

                        continue;
                    }



                    worksheet.Cell(newRow, 1).Value = marketDate;
                    worksheet.Cell(newRow, 2).Value = open;
                    worksheet.Cell(newRow, 3).Value = high;
                    worksheet.Cell(newRow, 4).Value = low;
                    worksheet.Cell(newRow, 5).Value = close;
                    worksheet.Cell(newRow, 6).Value = volume;
                }

                workbook.Save();
                Console.WriteLine("Workbook saved successfully.");
            }


            Console.WriteLine("All ticker updates completed.");
        }
    }
}