# YTEmpire Hardware Setup Documentation

## System Requirements

### Minimum Hardware Requirements
- **CPU**: AMD Ryzen 9 5900X or Intel Core i9-10900K
- **RAM**: 64GB DDR4 3200MHz
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **Storage**: 2TB NVMe SSD (Gen4)
- **Network**: 1Gbps Ethernet connection

### Recommended Hardware (Production)
- **CPU**: AMD Ryzen 9 9950X3D (16 cores, 32 threads)
- **RAM**: 128GB DDR5 6000MHz (4x32GB)
- **GPU**: NVIDIA RTX 5090 (32GB VRAM) or dual RTX 4090
- **Storage**: 
  - System: 1TB NVMe SSD Gen5
  - Data: 2x 2TB NVMe SSD Gen4 in RAID 0
  - Backup: 8TB HDD
- **Network**: 10Gbps Ethernet or 2.5Gbps minimum
- **PSU**: 1200W 80+ Platinum

## Operating System Installation

### Ubuntu 22.04 LTS Server Installation

1. **Download Ubuntu Server 22.04 LTS**
   ```bash
   wget https://releases.ubuntu.com/22.04/ubuntu-22.04.3-live-server-amd64.iso
   ```

2. **Create Bootable USB**
   - Use Rufus (Windows) or dd (Linux/Mac)
   - Boot from USB and follow installation wizard

3. **Partition Scheme**
   ```
   /boot/efi  - 512MB (EFI System Partition)
   /boot      - 2GB (ext4)
   /          - 100GB (ext4, root partition)
   /home      - 200GB (ext4, user data)
   /var       - 100GB (ext4, logs and Docker)
   /data      - Remaining space (ext4, application data)
   swap       - 32GB (swap partition)
   ```

4. **Initial System Configuration**
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install essential packages
   sudo apt install -y \
     build-essential \
     curl \
     wget \
     git \
     vim \
     htop \
     net-tools \
     software-properties-common \
     apt-transport-https \
     ca-certificates \
     gnupg \
     lsb-release
   
   # Configure timezone
   sudo timedatectl set-timezone America/New_York
   
   # Configure hostname
   sudo hostnamectl set-hostname ytempire-server
   ```

## NVIDIA GPU Setup

### Driver Installation

1. **Remove existing drivers**
   ```bash
   sudo apt-get purge nvidia* -y
   sudo apt-get autoremove -y
   sudo apt-get autoclean -y
   ```

2. **Add NVIDIA PPA and install drivers**
   ```bash
   # Add graphics drivers PPA
   sudo add-apt-repository ppa:graphics-drivers/ppa -y
   sudo apt update
   
   # Install NVIDIA driver 535 (RTX 5090 compatible)
   sudo apt install nvidia-driver-535 -y
   
   # Reboot
   sudo reboot
   ```

3. **Verify installation**
   ```bash
   nvidia-smi
   ```

### CUDA Toolkit Installation

1. **Install CUDA 12.2**
   ```bash
   # Add CUDA repository
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
   sudo dpkg -i cuda-keyring_1.0-1_all.deb
   sudo apt update
   
   # Install CUDA
   sudo apt install cuda-12-2 -y
   ```

2. **Configure environment**
   ```bash
   echo 'export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Install cuDNN**
   ```bash
   sudo apt install libcudnn8 libcudnn8-dev -y
   ```

## RAID Configuration

### Software RAID Setup (RAID 0 for Performance)

1. **Install mdadm**
   ```bash
   sudo apt install mdadm -y
   ```

2. **Create RAID array**
   ```bash
   # Identify drives
   lsblk
   
   # Create RAID 0 array (replace sdX and sdY with actual drives)
   sudo mdadm --create /dev/md0 --level=0 --raid-devices=2 /dev/nvme1n1 /dev/nvme2n1
   
   # Create filesystem
   sudo mkfs.ext4 /dev/md0
   
   # Mount RAID array
   sudo mkdir /data
   sudo mount /dev/md0 /data
   
   # Add to fstab for persistent mount
   echo '/dev/md0 /data ext4 defaults 0 0' | sudo tee -a /etc/fstab
   
   # Save RAID configuration
   sudo mdadm --detail --scan | sudo tee -a /etc/mdadm/mdadm.conf
   ```

## Network Configuration

### Static IP Configuration

1. **Edit netplan configuration**
   ```bash
   sudo nano /etc/netplan/00-installer-config.yaml
   ```

2. **Configure static IP**
   ```yaml
   network:
     ethernets:
       enp5s0:  # Replace with your interface name
         dhcp4: no
         addresses:
           - 192.168.1.100/24
         gateway4: 192.168.1.1
         nameservers:
           addresses:
             - 8.8.8.8
             - 8.8.4.4
     version: 2
   ```

3. **Apply configuration**
   ```bash
   sudo netplan apply
   ```

### Firewall Configuration

```bash
# Enable UFW
sudo ufw enable

# Allow SSH
sudo ufw allow 22/tcp

# Allow web services
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow application ports
sudo ufw allow 3000/tcp  # Frontend
sudo ufw allow 8000/tcp  # Backend API
sudo ufw allow 8001/tcp  # ML Server
sudo ufw allow 5678/tcp  # N8N
sudo ufw allow 5432/tcp  # PostgreSQL (restrict in production)
sudo ufw allow 6379/tcp  # Redis (restrict in production)

# Check status
sudo ufw status verbose
```

## Docker Installation

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install NVIDIA Docker support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install nvidia-docker2 -y
sudo systemctl restart docker

# Verify NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

## Performance Tuning

### System Optimization

1. **CPU Governor**
   ```bash
   # Install cpufrequtils
   sudo apt install cpufrequtils -y
   
   # Set to performance mode
   sudo cpufreq-set -g performance
   ```

2. **Swappiness**
   ```bash
   # Reduce swappiness for better performance
   echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
   sudo sysctl -p
   ```

3. **File Descriptors**
   ```bash
   # Increase file descriptor limits
   echo '* soft nofile 65535' | sudo tee -a /etc/security/limits.conf
   echo '* hard nofile 65535' | sudo tee -a /etc/security/limits.conf
   ```

### GPU Optimization

```bash
# Set GPU to maximum performance mode
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 450  # Set power limit (adjust for your GPU)

# Enable persistence mode
sudo nvidia-smi -pm ENABLED

# Set application clock
sudo nvidia-smi -ac 2100,1980  # Memory,Graphics (adjust for your GPU)
```

## Monitoring Setup

### System Monitoring Tools

```bash
# Install monitoring tools
sudo apt install -y \
  prometheus-node-exporter \
  netdata \
  iotop \
  iftop \
  nvtop

# Install NVIDIA GPU exporter for Prometheus
docker run -d \
  --name nvidia_gpu_exporter \
  --restart always \
  --gpus all \
  -p 9835:9835 \
  utkuozdemir/nvidia_gpu_exporter:1.1.0
```

### Log Management

```bash
# Configure log rotation
sudo nano /etc/logrotate.d/ytempire

# Add configuration:
/var/log/ytempire/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0640 www-data adm
    sharedscripts
    postrotate
        systemctl reload ytempire
    endscript
}
```

## Backup Configuration

### Automated Backup Script

```bash
#!/bin/bash
# Save as /usr/local/bin/ytempire-backup.sh

BACKUP_DIR="/backup"
DATE=$(date +%Y%m%d_%H%M%S)
DB_BACKUP="$BACKUP_DIR/db_$DATE.sql"
DATA_BACKUP="$BACKUP_DIR/data_$DATE.tar.gz"

# Database backup
docker exec ytempire-postgres pg_dumpall -U ytempire > $DB_BACKUP

# Data backup
tar -czf $DATA_BACKUP /data /var/lib/docker/volumes

# Keep only last 7 days
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

# Sync to remote (optional)
# rsync -avz $BACKUP_DIR/ backup@remote-server:/backups/
```

### Cron Setup

```bash
# Add to crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * /usr/local/bin/ytempire-backup.sh
```

## Security Hardening

### SSH Configuration

```bash
# Edit SSH config
sudo nano /etc/ssh/sshd_config

# Recommended settings:
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
Port 2222  # Change default port
AllowUsers ytempire  # Restrict to specific user

# Restart SSH
sudo systemctl restart sshd
```

### Fail2Ban Setup

```bash
# Install fail2ban
sudo apt install fail2ban -y

# Configure for SSH
sudo nano /etc/fail2ban/jail.local

[sshd]
enabled = true
port = 2222
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600

# Start fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

## Verification Checklist

- [ ] Ubuntu 22.04 LTS installed and updated
- [ ] Static IP configured
- [ ] Firewall configured with UFW
- [ ] NVIDIA drivers installed (verify with `nvidia-smi`)
- [ ] CUDA toolkit installed (verify with `nvcc --version`)
- [ ] Docker and Docker Compose installed
- [ ] NVIDIA Docker support working
- [ ] RAID array configured (if applicable)
- [ ] Monitoring tools installed
- [ ] Backup system configured
- [ ] Security hardening applied
- [ ] All services accessible from network

## Troubleshooting

### GPU Not Detected
```bash
# Check if GPU is recognized
lspci | grep -i nvidia

# Check kernel modules
lsmod | grep nvidia

# Rebuild NVIDIA modules
sudo dkms autoinstall
```

### Docker GPU Access Issues
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# Check Docker daemon config
cat /etc/docker/daemon.json
# Should contain: "default-runtime": "nvidia"
```

### Performance Issues
```bash
# Check system resources
htop
iotop
nvtop

# Check disk I/O
iostat -x 1

# Check network
iftop
```

## Support

For hardware-specific issues:
- NVIDIA Support: https://www.nvidia.com/en-us/support/
- Ubuntu Forums: https://ubuntuforums.org/
- Docker Documentation: https://docs.docker.com/

---

*Document Version: 1.0*
*Last Updated: 2024*
*Maintained by: YTEmpire Platform Operations Team*