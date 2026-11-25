using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using System.Globalization;

class Program
{
    static async Task Main()
    {
        string apiDate = "";
        Console.WriteLine("Choose Option:");
        Console.WriteLine("1 - Single Date");
        Console.WriteLine("2 - Date Range");
        Console.Write("Choice: ");

        string choice = Console.ReadLine();

        if (choice == "1")
        {
            Console.Write("Enter market date (MM-dd-yyyy): ");
            string inputDate = Console.ReadLine();

            DateTime marketDate = DateTime.ParseExact(
                inputDate,
                "MM-dd-yyyy",
                null);

            apiDate = marketDate.ToString("yyyy-MM-dd");

            Console.WriteLine($"Single Date Mode: {apiDate}");
        }
        else if (choice == "2")
        {
            Console.Write("Enter start date (MM-dd-yyyy): ");
            string startInput = Console.ReadLine();

            Console.Write("Enter end date (MM-dd-yyyy): ");
            string endInput = Console.ReadLine();

            DateTime startDate = DateTime.ParseExact(
                startInput,
                "MM-dd-yyyy",
                null);

            DateTime endDate = DateTime.ParseExact(
                endInput,
                "MM-dd-yyyy",
                null);
            DateTime currentDate = startDate;

            while (currentDate <= endDate)
            {
                if (currentDate.DayOfWeek != DayOfWeek.Saturday &&
                    currentDate.DayOfWeek != DayOfWeek.Sunday)
                {
                    string rangeApiDate =
                        currentDate.ToString("yyyy-MM-dd");

                    string rangeUrl =
    $"https://api.massive.com/v2/aggs/grouped/locale/us/market/stocks/{rangeApiDate}?adjusted=true&apiKey=***REMOVED***";

                    Console.WriteLine(
                        $"Downloading: {rangeApiDate}");

                    using (var client = new HttpClient())
                    {
                        try
                        {
                            var json = await client.GetStringAsync(rangeUrl);

                            var folder = @"C:\Massive";
                            Directory.CreateDirectory(folder);

                            var filePath =
                                Path.Combine(folder,
                                    $"prices_{rangeApiDate}.json");

                            File.WriteAllText(filePath, json);

                            Console.WriteLine(
                                $"Saved: prices_{rangeApiDate}.json");
                            await Task.Delay(2000);


                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine(
                                $"Failed: {rangeApiDate}");

                            Console.WriteLine(ex.Message);
                        }
                    }
                }

                currentDate = currentDate.AddDays(1);
            }

            Console.WriteLine(
                $"Range Mode: {startDate:yyyy-MM-dd} to {endDate:yyyy-MM-dd}");

            return;
        }
        else
        {
            Console.WriteLine("Invalid choice.");
            return;
        }

        var url =
            $"https://api.massive.com/v2/aggs/grouped/locale/us/market/stocks/{apiDate}?adjusted=true&apiKey=***REMOVED***";

        using (var client = new HttpClient())
        {
            try
            {
                Console.WriteLine("Downloading stock prices...");

                var json = await client.GetStringAsync(url);

                var folder = @"C:\Massive";
                Directory.CreateDirectory(folder);

                var filePath = Path.Combine(folder, "prices.json");
                File.WriteAllText(filePath, json);

                Console.WriteLine($"Saved JSON to: {filePath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error:");
                Console.WriteLine(ex.Message);
            }
        }
    }
}


