using System.ComponentModel.DataAnnotations.Schema;

namespace api_backend.Models
{
    [Table("cluster_changes")]
    public class ClusterChange
    {
        [Column("id")]
        public int Id { get; set; }

        [Column("currency_id")]
        public int CurrencyId { get; set; }

        [Column("symbol")]
        public string Symbol { get; set; } = string.Empty;

        [Column("from_cluster_id")]
        public int FromClusterId { get; set; }

        [Column("to_cluster_id")]
        public int ToClusterId { get; set; }

        [Column("change_timestamp")]
        public DateTime ChangeTimestamp { get; set; }
    }
}