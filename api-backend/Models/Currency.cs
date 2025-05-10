namespace api_backend.Models
{
    public class Currency
    {
        public int Id { get; set; }
        public string Symbol { get; set; } = string.Empty;
        public string? Name { get; set; }
        public int? ClusterId { get; set; }
        public DateTime CreatedAt { get; set; }
    }
}