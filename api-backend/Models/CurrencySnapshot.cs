using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json;

namespace api_backend.Models
{
    [Table("currency_snapshots")]
    public class CurrencySnapshot
    {
        [Key]
        [Column("id")]
        public int Id { get; set; }
        
        [Required]
        [Column("snapshot_time")]
        public DateTime SnapshotTime { get; set; }
        
        [Required]
        [Column("snapshot_data", TypeName = "jsonb")]
        public JsonDocument SnapshotData { get; set; } = null!;
    }

    public class CurrencySnapshotItem
    {
        public int Id { get; set; }
        public string Symbol { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public int ClusterId { get; set; }
    }
}
