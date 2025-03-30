import { CupSodaIcon as Cup } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function WalletConnect() {
  return (
    <div className="text-center max-w-md mx-auto px-4 py-12">
      <div className="bg-purple-100 w-20 h-20 rounded-full mx-auto flex items-center justify-center mb-6">
        <div className="relative">
          <Cup className="h-10 w-10 text-purple-600" />
          <div className="absolute -top-1 -right-1 flex gap-0.5">
            <div className="w-1.5 h-1.5 bg-purple-600 rounded-full"></div>
            <div className="w-1.5 h-1.5 bg-purple-600 rounded-full"></div>
            <div className="w-1.5 h-1.5 bg-purple-600 rounded-full"></div>
          </div>
        </div>
      </div>

      <h1 className="text-2xl font-semibold mb-3">Connect Your Wallet</h1>

      <p className="text-gray-600 mb-6">
        Connect your Solana wallet to create, mint, and manage tokens on the Solana blockchain.
      </p>

      <Button className="bg-purple-600 hover:bg-purple-700 text-white mb-8">Select Wallet</Button>

      <p className="text-sm text-gray-500">
        Note: This app uses the Solana Devnet. Make sure your wallet is configured for Devnet.
      </p>
    </div>
  )
}

