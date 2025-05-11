using System.ComponentModel.DataAnnotations.Schema;

namespace api_backend.Models
{
    [Table("currencies")]
    public class Currency
    {
        [Column("id")]
        public int Id { get; set; }
        
        [Column("symbol")]
        public string Symbol { get; set; } = string.Empty;
        
        [Column("name")]
        public string? Name { get; set; }
        
        [Column("cluster_id")]
        public int? ClusterId { get; set; }

        [Column("total_cluster_changes")]
        public int? TotalClusterChanges { get; set; }
        
        [Column("created_at")]
        public DateTime CreatedAt { get; set; }
    }
}
