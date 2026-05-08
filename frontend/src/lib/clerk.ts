export const clerkPublishableKey =
  process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY ?? "";

export function hasClerkConfig(): boolean {
  return clerkPublishableKey.length > 0;
}
