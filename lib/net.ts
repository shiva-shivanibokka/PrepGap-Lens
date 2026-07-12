// Block loopback / private / link-local / unique-local / cloud-metadata addresses (SSRF guard).
export function isPrivateIp(ip: string): boolean {
  const v = ip.replace(/^::ffff:/i, "");
  if (v.startsWith("127.") || v === "::1") return true;
  if (v.startsWith("10.") || v.startsWith("192.168.")) return true;
  if (v.startsWith("169.254.") || v.toLowerCase().startsWith("fe80:")) return true; // link-local + metadata
  if (/^172\.(1[6-9]|2\d|3[01])\./.test(v)) return true; // 172.16.0.0/12
  if (v === "0.0.0.0" || /^f[cd][0-9a-f]{2}:/i.test(v)) return true; // unspecified + unique-local IPv6
  return false;
}
