for host in $(cat workers.txt); do
  ssh -o "StrictHostKeyChecking no" $host uptime
done
